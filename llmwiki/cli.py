"""Command-line interface for the RAG core.

The CLI is intentionally small. Subcommands:

- ``llmwiki index``    — run one indexing pass over a vault.
- ``llmwiki status``   — show database state.
- ``llmwiki search``   — profile-aware dense/lexical/hybrid retrieval.
- ``llmwiki integrity`` — read-only projection diagnostics.
- ``llmwiki eval``     — golden-set evaluation of retrieval variants.

The plugin is the intended long-term entry point; the CLI is for
testing, scripting, evaluation, and one-off queries.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click

from .config import Settings
from .indexer import Indexer, summarise_database
from .logging import setup_logging


def _resolve_settings(
    *,
    vault: Path | None,
    db: Path | None,
    watch: bool,
    require_vault: bool = False,
) -> Settings:
    """Build Settings from CLI flags, falling back to env / defaults."""
    from .config import Settings as _Settings

    if require_vault and vault is None and not os.environ.get("LLMWIKI_VAULT"):
        raise click.UsageError("indexing requires --vault or LLMWIKI_VAULT; refusing to use cwd")
    base = _Settings.from_env()
    return _Settings(
        vault_path=(vault or base.vault_path).expanduser().resolve(),
        db_path=(db or base.db_path).expanduser().resolve(),
        file_watch=watch or base.file_watch,
        log_level=base.log_level,
        log_format=base.log_format,
    )


@click.group()
@click.option("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR (overrides env)")
@click.option(
    "--log-format", default=None, type=click.Choice(["text", "json"]), help="Log output format"
)
def main(log_level: str | None, log_format: str | None) -> None:
    """Local-first hybrid RAG over an Obsidian vault."""
    setup_logging(level=log_level, fmt=log_format)


@main.command()
@click.option(
    "--vault",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Path to the Obsidian vault (required unless $LLMWIKI_VAULT is set)",
)
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite database file (default: $LLMWIKI_DB or ./.data/llmwiki.sqlite)",
)
@click.option(
    "--mode",
    type=click.Choice(["incremental", "full"]),
    default="incremental",
    help="Indexing mode (default: incremental)",
)
@click.option(
    "--watch/--no-watch",
    default=False,
    help="After indexing, keep watching the vault and re-index on changes (coalesced)",
)
@click.option(
    "--debounce",
    default=2.0,
    show_default=True,
    type=float,
    help="Seconds of quiet before a watch-triggered reindex",
)
@click.option(
    "--embed/--no-embed",
    default=True,
    help="Generate embeddings for chunks (Phase 3; default: enabled)",
)
def index(
    vault: Path | None,
    db: Path | None,
    mode: str,
    watch: bool,
    debounce: float,
    embed: bool,
) -> None:
    """Index a vault into the local SQLite database.

    On first run against a vault with no embeddings this will embed
    every chunk (one-time cost). A 5000-chunk vault typically takes
    30-45 minutes on a multi-core box; the work is resumable (kill
    and re-run safely). After that, unchanged reindexes are sub-second
    and search queries are ~30ms end-to-end.
    """
    settings = _resolve_settings(vault=vault, db=db, watch=False, require_vault=True)
    click.echo(f"indexing: vault={settings.vault_path} db={settings.db_path} mode={mode}")
    embedder = None
    if embed:
        # Lazy import: FastEmbed is heavy and may not be installed.
        from . import db as dbmod
        from .embeddings import FastEmbedEmbedder

        click.echo(f"embedder: {settings.embedding_model}")
        # Pre-flight: warn if a cold-start backfill is likely. We
        # only emit the warning when embeddings ARE enabled and the
        # DB has chunks without vectors — that's the cold-start case.
        if settings.db_path.exists():
            with dbmod.connect(settings.db_path) as conn:
                dbmod.init_schema(conn)
                n_chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
                # chunk_embeddings may not exist in a v2 DB; ignore errors.
                try:
                    n_emb = int(conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0])
                except Exception:
                    n_emb = 0
                if n_chunks > 0 and n_emb < n_chunks:
                    pending = n_chunks - n_emb
                    click.echo(
                        f"cold start: {pending} chunk(s) need embedding; "
                        f"this run will take several minutes. "
                        f"Use --no-embed to skip and embed later.",
                        err=True,
                    )
        embedder = FastEmbedEmbedder(model_name=settings.embedding_model)
    else:
        click.echo("embeddings: disabled (--no-embed)", err=True)
    # The vector store needs a SQLite connection. We open it inside
    # ``Indexer.run`` (one connection per run) so the store's lifetime
    # is naturally bounded by the run. The CLI only owns the
    # embedder; the store is constructed lazily.
    if watch:
        from .watch import watch_vault

        if mode == "full":
            Indexer(settings, embedder=embedder).run(mode="full")
        click.echo("watching for changes; press Ctrl-C to stop", err=True)
        try:
            runs = watch_vault(settings, embedder=embedder, debounce_s=debounce, initial_run=True)
        except KeyboardInterrupt:
            runs = -1
        click.echo(f"watch stopped after {runs} run(s)" if runs >= 0 else "watch stopped", err=True)
        return
    indexer = Indexer(settings, embedder=embedder)
    stats = indexer.run(mode=mode)
    click.echo(
        f"done: seen={stats.documents_seen} added={stats.documents_added} "
        f"updated={stats.documents_updated} removed={stats.documents_removed} "
        f"skipped={stats.documents_skipped} "
        f"chunks: +{stats.chunks_added} ~{stats.chunks_updated} -{stats.chunks_removed} "
        f"embeddings: built={stats.embeddings_built} rebuilt={stats.embeddings_rebuilt} "
        f"errors={len(stats.errors)}"
    )
    if stats.errors:
        sys.exit(1)


@main.command()
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite database file (default: $LLMWIKI_DB or ./.data/llmwiki.sqlite)",
)
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON")
def status(db: Path | None, as_json: bool) -> None:
    """Show database state."""
    from .config import Settings as _Settings

    base = _Settings.from_env()
    db_path = (db or base.db_path).expanduser().resolve()
    summary = summarise_database(db_path)
    if as_json:
        click.echo(json.dumps(summary, indent=2, default=str))
        return
    if not summary.get("exists"):
        click.echo(f"database does not exist: {db_path}")
        return
    click.echo(f"database: {summary['path']}")
    click.echo(f"documents: {summary['documents']}")
    click.echo(f"chunks: {summary['chunks']}")
    click.echo(f"embeddings: {summary['embeddings']}")
    click.echo(f"index runs: {summary['runs']}")
    last_run = summary.get("last_run")
    if last_run is not None:
        # last_run is dict[str, object]; cast to dict[str, Any] for safe subscripting.
        from typing import Any, cast

        lr = cast("dict[str, Any]", last_run)
        click.echo(
            f"last run: mode={lr['mode']} added={lr['added']} updated={lr['updated']} "
            f"removed={lr['removed']} skipped={lr['skipped']}"
        )


@main.command()
@click.option(
    "--vault", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None
)
@click.option("--db", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON")
def integrity(vault: Path | None, db: Path | None, as_json: bool) -> None:
    """Check the retrieval projection for orphan, stale, or mixed rows."""
    from . import db as dbmod

    settings = _resolve_settings(vault=vault, db=db, watch=False, require_vault=True)
    report = dbmod.inspect_integrity(settings.db_path, vault_path=settings.vault_path)
    if as_json:
        click.echo(json.dumps(report, indent=2, default=str))
    elif not report["exists"]:
        click.echo(f"database does not exist: {settings.db_path}")
    else:
        click.echo(f"database: {report['path']}")
        click.echo(f"schema version: {report['schema_version']}")
        click.echo(f"rebuild state: {report['rebuild_state']}")
        click.echo(f"orphan vectors: {report['orphan_vectors']}")
        click.echo(f"orphan chunks: {report['orphan_chunks']}")
        click.echo(f"chunks without embeddings: {report['chunks_without_embeddings']}")
        missing_on_disk = report["documents_missing_on_disk"]
        assert isinstance(missing_on_disk, list)
        click.echo(f"documents missing on disk: {len(missing_on_disk)}")
        click.echo(f"mixed embedding models: {report['mixed_embedding_models']}")
    if not report["ok"]:
        raise click.ClickException("projection integrity check failed")


@main.command()
@click.option("--db", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--query", required=True, help="Search query")
@click.option("--top-k", default=10, show_default=True, type=int)
@click.option(
    "--profile",
    default="answer",
    show_default=True,
    help="Corpus profile: answer, evidence, history, all, or project:<id>",
)
@click.option(
    "--mode",
    default=None,
    type=click.Choice(["dense", "lexical", "hybrid"]),
    help="Retrieval channels (default: configured retrieval_mode)",
)
@click.option("--rerank/--no-rerank", default=None, help="Force the cross-encoder on or off")
@click.option("--since", default=None, help="Only pages modified on/after this date (YYYY-MM-DD)")
@click.option("--graph/--no-graph", default=None, help="Force the linked-pages channel on or off")
@click.option(
    "--multiquery/--no-multiquery", default=None, help="Split multi-part questions and fuse"
)
@click.option("--context", "as_context", is_flag=True, help="Print the budgeted LLM context block")
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON")
def search(
    db: Path | None,
    query: str,
    top_k: int,
    profile: str,
    mode: str | None,
    rerank: bool | None,
    since: str | None,
    graph: bool | None,
    multiquery: bool | None,
    as_context: bool,
    as_json: bool,
) -> None:
    """Profile-aware retrieval over the indexed vault (dense, lexical, or hybrid)."""
    updated_after_ns: int | None = None
    if since:
        from datetime import datetime

        try:
            updated_after_ns = int(datetime.fromisoformat(since).timestamp() * 1_000_000_000)
        except ValueError as exc:
            raise click.BadParameter("--since expects YYYY-MM-DD") from exc
    from . import db as dbmod
    from .retrieval import Retriever, context_for

    base = Settings.from_env()
    db_path = (db or base.db_path).expanduser().resolve()
    effective_mode = mode or base.retrieval_mode
    embedder = None
    if effective_mode in ("dense", "hybrid"):
        from .embeddings import FastEmbedEmbedder

        embedder = FastEmbedEmbedder(model_name=base.embedding_model)
    reranker = None
    if rerank or (rerank is None and base.reranker_enabled):
        from .reranker import FastEmbedReranker

        reranker = FastEmbedReranker(model_name=base.reranker_model)
    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        retriever = Retriever(conn, embedder=embedder, settings=base, reranker=reranker)
        if effective_mode != "lexical":
            from .vector import SqliteVecStore

            if SqliteVecStore(conn).count() == 0:
                click.echo(
                    f"no embeddings found in {db_path}; run `llmwiki index` first.", err=True
                )
                sys.exit(2)
        result = retriever.retrieve(
            query,
            profile=profile,
            mode=effective_mode,
            top_k=top_k,
            rerank=bool(reranker) if rerank is None else rerank,
            updated_after_ns=updated_after_ns,
            graph_channel=graph,
            multiquery=multiquery,
        )
    if as_context:
        block = context_for(result, base)
        if as_json:
            click.echo(json.dumps(block.to_dict(), indent=2, ensure_ascii=False))
        else:
            click.echo(block.text if not block.empty else "no results.")
        return
    if as_json:
        payload = {
            "query": result.query,
            "profile": result.profile,
            "mode": result.mode,
            "intent": result.intent,
            "conflicts": list(result.conflicts),
            "elapsed_ms": round(result.elapsed_ms, 2),
            "results": [
                {
                    "chunk_id": c.chunk_id,
                    "path": c.path,
                    "title": c.title,
                    "heading_path": list(c.heading_path),
                    "section_name": c.section_name,
                    "position": c.position,
                    "text": c.text,
                    "authority_class": c.authority_class,
                    "authority_match": c.authority_match,
                    "dense_rank": c.dense_rank,
                    "dense_distance": c.dense_distance,
                    "lexical_rank": c.lexical_rank,
                    "bm25_score": c.bm25_score,
                    "rrf_score": c.rrf_score,
                    "rerank_score": c.rerank_score,
                }
                for c in result.candidates
            ],
        }
        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if not result.candidates:
        click.echo("no results.")
        return
    click.echo(
        f"mode={result.mode} profile={result.profile} intent={result.intent} "
        f"({result.elapsed_ms:.0f} ms)"
    )
    for label in result.conflicts:
        click.echo(f"conflict: {label}")
    for c in result.candidates:
        metrics = []
        if c.dense_distance is not None:
            metrics.append(f"d={c.dense_distance:.3f}")
        if c.bm25_score is not None:
            metrics.append(f"bm25={c.bm25_score:.2f}")
        if c.rrf_score is not None:
            metrics.append(f"rrf={c.rrf_score:.4f}")
        if c.rerank_score is not None:
            metrics.append(f"rr={c.rerank_score:.2f}")
        flag = "*" if c.authority_match else " "
        click.echo(
            f"[{' '.join(metrics)}]{flag} {c.path}#{c.section_name or '(intro)'} "
            f"(chunk {c.position}, {c.authority_class})"
        )
        snippet = c.text.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        click.echo(f"    {snippet}")
        click.echo("")


@main.command()
@click.option(
    "--vault", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None
)
@click.option("--db", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--queries", default=20, show_default=True, type=int, help="Queries per mode")
@click.option("--json", "as_json", is_flag=True)
def bench(vault: Path | None, db: Path | None, queries: int, as_json: bool) -> None:
    """Measure no-change reindex time, retrieval latency per mode, and peak RSS."""
    import resource
    import statistics
    import time as _time

    from . import db as dbmod
    from .embeddings import FastEmbedEmbedder
    from .retrieval import Retriever

    settings = _resolve_settings(vault=vault, db=db, watch=False, require_vault=True)
    t0 = _time.perf_counter()
    embedder = FastEmbedEmbedder(model_name=settings.embedding_model)
    _ = embedder.dim
    model_load_s = _time.perf_counter() - t0
    t0 = _time.perf_counter()
    stats = Indexer(settings, embedder=embedder).run(mode="incremental")
    reindex_s = _time.perf_counter() - t0
    with dbmod.connect(settings.db_path) as conn:
        titles = [
            str(r[0])
            for r in conn.execute(
                "SELECT title FROM documents WHERE source_kind = 'wiki' ORDER BY id LIMIT ?",
                (max(queries, 1),),
            ).fetchall()
        ]
        retriever = Retriever(conn, embedder=embedder, settings=settings)
        latencies: dict[str, dict[str, float]] = {}
        for mode in ("dense", "lexical", "hybrid"):
            samples: list[float] = []
            for title in titles:
                t0 = _time.perf_counter()
                retriever.retrieve(f"what does the wiki say about {title}?", mode=mode, top_k=10)
                samples.append((_time.perf_counter() - t0) * 1000.0)
            samples.sort()
            latencies[mode] = {
                "p50_ms": round(statistics.median(samples), 1) if samples else 0.0,
                "p95_ms": round(
                    samples[int(len(samples) * 0.95) - 1] if len(samples) > 1 else samples[0], 1
                )
                if samples
                else 0.0,
                "n": float(len(samples)),
            }
        counts = {
            "documents": int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]),
            "chunks": int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]),
        }
    report = {
        "model_load_s": round(model_load_s, 2),
        "reindex_no_change_s": round(reindex_s, 2),
        "reindex_seen": stats.documents_seen,
        "latency": latencies,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "counts": counts,
    }
    if as_json:
        click.echo(json.dumps(report, indent=2))
        return
    click.echo(
        f"model load: {report['model_load_s']} s; no-change reindex: {report['reindex_no_change_s']} s "
        f"({stats.documents_seen} docs); peak RSS {report['peak_rss_mb']} MB"
    )
    for mode, lat in latencies.items():
        click.echo(
            f"  {mode:8} p50={lat['p50_ms']:.0f} ms p95={lat['p95_ms']:.0f} ms (n={int(lat['n'])})"
        )


@main.command()
@click.option(
    "--vault", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None
)
@click.option("--db", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--profile", "default_profile", default="answer", show_default=True)
@click.option("--max-results", default=6, show_default=True, type=int)
@click.option("--allow-full-rebuild", is_flag=True, help="Permit mode=full reindex over MCP")
@click.option("--watch", is_flag=True, help="Keep the projection fresh while the server runs")
def mcp(
    vault: Path | None,
    db: Path | None,
    default_profile: str,
    max_results: int,
    allow_full_rebuild: bool,
    watch: bool,
) -> None:
    """Serve llmwiki_search / llmwiki_status / llmwiki_reindex over MCP (stdio transport)."""
    from .mcp_server import serve_stdio
    from .service import ServiceConfig

    settings = _resolve_settings(vault=vault, db=db, watch=False, require_vault=True)
    config = ServiceConfig(
        vault=str(settings.vault_path),
        db=str(settings.db_path),
        default_profile=default_profile,
        max_results=max_results,
        allow_full_rebuild=allow_full_rebuild,
        watch=watch,
    )
    serve_stdio(config)


@main.command()
@click.argument("path")
@click.option("--db", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--limit", default=20, show_default=True, type=int)
@click.option("--json", "as_json", is_flag=True)
def related(path: str, db: Path | None, limit: int, as_json: bool) -> None:
    """Pages related to PATH by links, backlinks, title mentions and community."""
    from . import db as dbmod
    from .entities import related_pages

    base = Settings.from_env()
    db_path = (db or base.db_path).expanduser().resolve()
    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        pages = related_pages(conn, path.strip().lstrip("/"), limit=limit)
    if as_json:
        click.echo(json.dumps([p.__dict__ for p in pages], indent=2, ensure_ascii=False))
        return
    if not pages:
        click.echo("no related pages (unknown path or isolated page).")
        return
    for p in pages:
        click.echo(f"{p.weight:3}  {p.relation:15} {p.path}  ({p.title})")


@main.command()
@click.option("--db", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--limit", default=10, show_default=True, type=int)
def communities(db: Path | None, limit: int) -> None:
    """Largest link/mention communities in the projection."""
    from . import db as dbmod
    from .entities import community_summary

    base = Settings.from_env()
    db_path = (db or base.db_path).expanduser().resolve()
    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        for c in community_summary(conn, limit=limit):
            titles = c["sample_titles"]
            sample = ", ".join(str(t) for t in titles) if isinstance(titles, list) else ""
            click.echo(f"community {c['community_id']:5} size={c['size']:4}  {sample}")


@main.group()
def eval() -> None:
    """Golden-set evaluation of retrieval variants."""


@eval.command("validate")
@click.option(
    "--set",
    "golden_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--vault", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None
)
def eval_validate(golden_path: Path, vault: Path | None) -> None:
    """Validate a golden question set (schema, paths, headings, stratification)."""
    from .evaluation.golden import load_golden, stratification_report, validate_golden

    settings = _resolve_settings(vault=vault, db=None, watch=False, require_vault=True)
    golden = load_golden(golden_path)
    problems = validate_golden(golden, vault=settings.vault_path) + stratification_report(golden)
    click.echo(f"questions: {len(golden.questions)}")
    for category, counts in golden.counts().items():
        click.echo(f"  {category}: dev={counts['dev']} heldout={counts['heldout']}")
    for problem in problems:
        click.echo(f"problem: {problem}", err=True)
    if problems:
        raise click.ClickException(f"{len(problems)} problem(s) found")
    click.echo("ok")


@eval.command("run")
@click.option(
    "--set",
    "golden_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--vault", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None
)
@click.option("--db", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option(
    "--variant",
    "variants",
    multiple=True,
    type=click.Choice(["dense", "lexical", "hybrid", "hybrid+rerank"]),
    default=("dense",),
    show_default=True,
)
@click.option(
    "--split", type=click.Choice(["dev", "heldout", "all"]), default="heldout", show_default=True
)
@click.option("--top-k", default=10, show_default=True, type=int)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("evals/runs"),
    show_default=True,
)
@click.option("--no-outcomes", is_flag=True, help="Omit per-question outcomes from the record")
def eval_run(
    golden_path: Path,
    vault: Path | None,
    db: Path | None,
    variants: tuple[str, ...],
    split: str,
    top_k: int,
    out_dir: Path,
    no_outcomes: bool,
) -> None:
    """Run one or more retrieval variants over a golden set and record the results."""
    from . import db as dbmod
    from .evaluation.golden import load_golden, validate_golden
    from .evaluation.runner import format_comparison, run_variant, write_run
    from .retrieval import Retriever

    settings = _resolve_settings(vault=vault, db=db, watch=False, require_vault=True)
    golden = load_golden(golden_path)
    problems = validate_golden(golden, vault=settings.vault_path)
    if problems:
        raise click.ClickException(f"golden set invalid: {problems[0]} (+{len(problems) - 1} more)")
    needs_dense = any(v != "lexical" for v in variants)
    embedder = None
    if needs_dense:
        from .embeddings import FastEmbedEmbedder

        embedder = FastEmbedEmbedder(model_name=settings.embedding_model)
    reranker = None
    if any(v == "hybrid+rerank" for v in variants):
        from .reranker import FastEmbedReranker

        reranker = FastEmbedReranker(model_name=settings.reranker_model)
    snapshot = {
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model if reranker else None,
        "retrieval_top_k_dense": settings.retrieval_top_k_dense,
        "retrieval_top_k_lexical": settings.retrieval_top_k_lexical,
        "rrf_k": settings.rrf_k,
        "max_chunks_per_document": settings.max_chunks_per_document,
        "rerank_candidates": settings.rerank_candidates,
    }
    records = []
    with dbmod.connect(settings.db_path) as conn:
        dbmod.init_schema(conn)
        retriever = Retriever(conn, embedder=embedder, settings=settings, reranker=reranker)
        for variant in variants:
            record = run_variant(
                golden,
                variant=variant,
                split=None if split == "all" else split,
                retriever=retriever,
                conn=conn,
                settings_snapshot=snapshot,
                vault=settings.vault_path,
                top_k=top_k,
                include_outcomes=not no_outcomes,
            )
            path = write_run(record, out_dir)
            click.echo(f"wrote {path}", err=True)
            records.append(record.to_dict())
    click.echo(format_comparison(records))


@eval.command("calibrate")
@click.option(
    "--set",
    "golden_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--vault", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None
)
@click.option("--db", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("hermes_plugin/injection_gate.json"),
    show_default=True,
)
@click.option("--top-k", default=10, show_default=True, type=int)
def eval_calibrate(
    golden_path: Path, vault: Path | None, db: Path | None, out_path: Path, top_k: int
) -> None:
    """Fit the automatic-injection gate on dev, measure Gate A on held-out, write the gate file."""
    from . import db as dbmod
    from .embeddings import FastEmbedEmbedder
    from .evaluation.calibration import calibrate, write_gate
    from .evaluation.golden import load_golden, validate_golden
    from .evaluation.runner import git_sha
    from .retrieval import Retriever

    settings = _resolve_settings(vault=vault, db=db, watch=False, require_vault=True)
    golden = load_golden(golden_path)
    problems = validate_golden(golden, vault=settings.vault_path)
    if problems:
        raise click.ClickException(f"golden set invalid: {problems[0]} (+{len(problems) - 1} more)")
    embedder = FastEmbedEmbedder(model_name=settings.embedding_model)
    with dbmod.connect(settings.db_path) as conn:
        dbmod.init_schema(conn)
        projects = [
            str(r[0])
            for r in conn.execute(
                "SELECT DISTINCT project_id FROM documents WHERE project_id IS NOT NULL"
            ).fetchall()
        ]
        retriever = Retriever(conn, embedder=embedder, settings=settings)
        gate, metrics = calibrate(
            golden,
            lambda q: retriever.retrieve(q.query, profile=q.profile, mode="hybrid", top_k=top_k),
            known_projects=projects,
            fitted_on=f"{golden.version}@{git_sha()}",
        )
    write_gate(gate, out_path)
    click.echo(f"wrote {out_path}", err=True)
    click.echo(json.dumps({k: v for k, v in metrics.items() if k != "routing"}, indent=2))
    click.echo(json.dumps(metrics["routing"], indent=2))
    if not metrics["gate_a_passed"]:
        click.echo(
            "Gate A NOT passed: automatic injection stays uncertified (opt-in only).", err=True
        )


@eval.command("regress")
@click.argument("baseline", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("candidate", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def eval_regress(baseline: Path, candidate: Path) -> None:
    """Apply the docs/evaluation.md regression rule to two run records (exit 1 on regression)."""
    from .evaluation.runner import load_run, regression_report

    report = regression_report(load_run(baseline), load_run(candidate))
    click.echo(json.dumps(report, indent=2))
    if report["comparable"] is False:
        raise click.ClickException("runs are not comparable: " + "; ".join(report["problems"]))
    if report["regression"]:
        raise click.ClickException("regression: " + "; ".join(report["problems"]))


@eval.command("report")
@click.option(
    "--runs",
    "runs_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("evals/runs"),
    show_default=True,
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("docs/benchmarks.md"),
    show_default=True,
)
def eval_report(runs_dir: Path, out_path: Path) -> None:
    """Render docs/benchmarks.md from recorded runs (latest per golden/split/variant)."""
    from .evaluation.runner import format_benchmark_markdown, load_run

    runs = [load_run(p) for p in sorted(runs_dir.glob("*.json"))]
    if not runs:
        raise click.ClickException(f"no run records in {runs_dir}")
    out_path.write_text(format_benchmark_markdown(runs), encoding="utf-8")
    click.echo(f"wrote {out_path} from {len(runs)} run(s)")


@eval.command("compare")
@click.argument(
    "runs", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--category", default=None, help="Show one category instead of overall")
def eval_compare(runs: tuple[Path, ...], category: str | None) -> None:
    """Print a comparison table for recorded evaluation runs."""
    from .evaluation.runner import format_comparison, load_run

    click.echo(format_comparison([load_run(p) for p in runs], category=category))


if __name__ == "__main__":  # pragma: no cover
    main()
