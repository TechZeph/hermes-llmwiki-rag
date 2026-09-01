"""Command-line interface for the RAG core.

The CLI is intentionally small. Subcommands:

- ``llmwiki index``    — run one indexing pass over a vault.
- ``llmwiki status``   — show database state.
- ``llmwiki search``   — (Phase 5+) search the indexed vault.

Phase 1 implements ``index`` and ``status`` only. The plugin is the
intended long-term entry point; the CLI is for testing, scripting,
and one-off queries.
"""

from __future__ import annotations

import json
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
) -> Settings:
    """Build Settings from CLI flags, falling back to env / defaults."""
    from .config import Settings as _Settings

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
    help="Path to the Obsidian vault (default: $LLMWIKI_VAULT or cwd)",
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
    help="Watch the vault for changes and re-index (Phase 1: not yet implemented)",
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
    embed: bool,
) -> None:
    """Index a vault into the local SQLite database.

    On first run against a vault with no embeddings this will embed
    every chunk (one-time cost). A 5000-chunk vault typically takes
    30-45 minutes on a multi-core box; the work is resumable (kill
    and re-run safely). After that, unchanged reindexes are sub-second
    and search queries are ~30ms end-to-end.
    """
    if watch:
        click.echo("--watch is not yet implemented (Phase 1); running one pass.", err=True)
    settings = _resolve_settings(vault=vault, db=db, watch=False)
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
                n_chunks = int(
                    conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                )
                # chunk_embeddings may not exist in a v2 DB; ignore errors.
                try:
                    n_emb = int(
                        conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
                    )
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

    base = Settings.from_env()
    db_path = (db or base.db_path).expanduser().resolve()
    vault_path = (vault or base.vault_path).expanduser().resolve()
    report = dbmod.inspect_integrity(db_path, vault_path=vault_path)
    if as_json:
        click.echo(json.dumps(report, indent=2, default=str))
    elif not report["exists"]:
        click.echo(f"database does not exist: {db_path}")
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
@click.option(
    "--vault", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None
)
@click.option("--db", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--query", required=True, help="Search query")
@click.option("--top-k", default=10, show_default=True, type=int)
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON")
def search(
    vault: Path | None,
    db: Path | None,
    query: str,
    top_k: int,
    as_json: bool,
) -> None:
    """Semantic search over the indexed vault (Phase 3: vector only).

    Embeds ``query`` with the configured FastEmbed model and ranks
    the top-K most similar chunks by cosine similarity. Hybrid
    lexical+dense fusion is Phase 5+.
    """
    from . import db as dbmod
    from .embeddings import FastEmbedEmbedder
    from .vector import SqliteVecStore

    base = Settings.from_env()
    db_path = (db or base.db_path).expanduser().resolve()
    embedder = FastEmbedEmbedder(model_name=base.embedding_model)
    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        store = SqliteVecStore(conn)
        if store.count() == 0:
            click.echo(
                f"no embeddings found in {db_path}; "
                f"run `llmwiki index` first.",
                err=True,
            )
            sys.exit(2)
        q_vec = embedder.embed([query])[0]
        hits = store.search(q_vec, top_k=top_k)
        if not hits:
            click.echo("no results.")
            return
        # Hydrate chunk -> document for human-readable output.
        ids = [cid for cid, _ in hits]
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"""
            SELECT c.id, c.document_id, c.position, c.section_name, c.text,
                   d.path, d.title
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        by_id = {int(r[0]): r for r in rows}
        results = []
        for cid, distance in hits:
            row = by_id.get(cid)
            if row is None:
                continue
            results.append(
                {
                    "chunk_id": int(row[0]),
                    "document_id": int(row[1]),
                    "position": int(row[2]),
                    "section_name": str(row[3]),
                    "text": str(row[4]),
                    "path": str(row[5]),
                    "title": str(row[6]),
                    "distance": float(distance),
                }
            )
    if as_json:
        click.echo(json.dumps(results, indent=2, ensure_ascii=False))
        return
    for r in results:
        click.echo(
            f"[d={r['distance']:.4f}] {r['path']}#{r['section_name'] or '(intro)'} "
            f"(chunk {r['position']})"
        )
        # Show first 240 chars of the chunk text.
        snippet = str(r["text"]).strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        click.echo(f"    {snippet}")
        click.echo("")


if __name__ == "__main__":  # pragma: no cover
    main()
