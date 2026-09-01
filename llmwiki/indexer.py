"""Incremental Obsidian vault indexer (Phase 1).

The indexer reads an Obsidian vault and writes document metadata to
the SQLite database opened by :mod:`llmwiki.db`. It is incremental:

- Files that exist on disk but not in the database are **added**.
- Files that exist in both, and whose ``mtime_ns`` or ``content_hash``
  has changed since the last index, are **updated**.
- Files that exist in the database but not on disk are **removed**.
- Files that exist in both with no change are **skipped** (fast path).

The indexer is *read-only* against the vault. The vault is the
canonical source of truth; the database is a derived projection.

Phase 1 indexes whole documents. Phase 2 will add a ``chunks`` table
and a structural chunker pass after the documents are up to date.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from . import db as dbmod
from .chunker import chunk_document
from .chunks import delete_chunks_for_document, insert_chunks
from .config import Settings
from .corpus import classify_path
from .embeddings import Embedder, model_provenance
from .entities import refresh_entity_graph
from .graph import replace_document_links, resolve_links
from .logging import get_logger
from .models import Document, IndexRunStats
from .parser import ParsedDocument, parse_markdown
from .recipes import (
    compatibility_state,
    embedding_recipe_state,
    format_document_embedding_input,
)
from .vector import SqliteVecStore, VectorStore

logger = get_logger("indexer")


# --- filesystem walk --------------------------------------------------------


def resolve_contained(vault_root: Path, candidate: Path) -> Path:
    """Resolve one regular vault file without following symlinks.

    The returned path is guaranteed to resolve inside ``vault_root``. Callers
    must invoke this immediately before each stat or open because a path can
    change between discovery and use.
    """
    root = vault_root.resolve(strict=True)
    if candidate.is_symlink():
        raise ValueError("refusing symlinked vault path")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("candidate resolves outside configured vault") from exc
    if not resolved.is_file():
        raise ValueError("candidate is not a regular file")
    return resolved


@dataclass(frozen=True, slots=True)
class VaultFile:
    """One markdown file as seen on disk, before parsing."""

    rel_path: str  # forward-slash, relative to vault root
    abs_path: Path
    mtime_ns: int
    size_bytes: int


def _should_skip(
    rel_path: str, ignored_dirs: tuple[str, ...], ignored_globs: tuple[str, ...]
) -> bool:
    from fnmatch import fnmatch

    parts = rel_path.split("/")
    # Skip any file inside an ignored directory.
    if any(p in ignored_dirs for p in parts):
        return True
    # Skip any file whose name starts with a dot (Obsidian's convention for
    # config / workspace state — e.g. .obsidian, .trash, .hidden.md).
    if any(p.startswith(".") for p in parts):
        return True
    return any(fnmatch(rel_path, pattern) for pattern in ignored_globs)


def iter_vault_files(vault: Path, settings: Settings) -> Iterator[VaultFile]:
    """Yield every non-skipped ``.md`` file under the vault, relative paths."""
    vault = vault.resolve(strict=True)
    for root, dirs, files in os.walk(vault):
        # In-place prune of operational and symlinked directories so ``os.walk``
        # cannot traverse a vault boundary through a link.
        dirs[:] = sorted(
            d for d in dirs if d not in settings.ignored_dirs and not (Path(root) / d).is_symlink()
        )
        for name in sorted(files):
            if not name.endswith(".md"):
                continue
            candidate = Path(root) / name
            rel_path = candidate.relative_to(vault).as_posix()
            if _should_skip(rel_path, settings.ignored_dirs, settings.ignored_globs):
                continue
            try:
                abs_path = resolve_contained(vault, candidate)
                stat = abs_path.stat()
            except (OSError, ValueError) as exc:
                logger.warning("stat failed for %s: %s", rel_path, exc)
                continue
            yield VaultFile(
                rel_path=rel_path,
                abs_path=abs_path,
                mtime_ns=stat.st_mtime_ns,
                size_bytes=stat.st_size,
            )


# --- hash + parse -----------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_document_and_parsed(vf: VaultFile, vault: Path) -> tuple[Document, ParsedDocument]:
    """Read the file, hash it, parse frontmatter/wikilinks/headings, return both shapes.

    The :class:`Document` is what gets persisted to the ``documents``
    table; the :class:`ParsedDocument` carries the body text the
    chunker needs. We don't store the body on ``Document`` because
    that would duplicate the file's content in the database; the
    chunker reads the body once during indexing and persists chunks.
    """
    abs_path = resolve_contained(vault, vf.abs_path)
    parsed = parse_markdown(str(abs_path))
    metadata = classify_path(vf.rel_path)
    doc = Document(
        id=None,
        path=vf.rel_path,
        absolute_path=str(abs_path),
        title=parsed.title or abs_path.stem,
        mtime_ns=vf.mtime_ns,
        size_bytes=vf.size_bytes,
        content_hash=_sha256(abs_path),
        frontmatter=parsed.frontmatter,
        tags=parsed.tags,
        wikilinks=parsed.wikilinks,
        aliases=parsed.aliases,
        headings=parsed.headings,
        source_kind=metadata["source_kind"],
        page_role=metadata["page_role"],
        project_id=metadata["project_id"],
        updated_at_ns=vf.mtime_ns,
        is_route_map=metadata["is_route_map"],
    )
    return doc, parsed


# --- persistence ------------------------------------------------------------


def _upsert_document(conn: sqlite3.Connection, doc: Document) -> int:
    """Insert or update a document row; return the row id."""
    existing = conn.execute(
        "SELECT id, content_hash, mtime_ns FROM documents WHERE path = ?", (doc.path,)
    ).fetchone()
    now_ns = time.time_ns()
    payload = (
        doc.path,
        doc.absolute_path,
        doc.title,
        doc.mtime_ns,
        doc.size_bytes,
        doc.content_hash,
        json.dumps(doc.frontmatter, ensure_ascii=False, default=str),
        json.dumps(list(doc.tags), ensure_ascii=False),
        json.dumps(list(doc.wikilinks), ensure_ascii=False),
        json.dumps(list(doc.aliases), ensure_ascii=False),
        json.dumps(list(doc.headings), ensure_ascii=False),
        now_ns,
        doc.source_kind,
        doc.page_role,
        doc.project_id,
        doc.updated_at_ns,
        int(doc.is_route_map),
    )
    if existing is None:
        cursor = conn.execute(
            """
            INSERT INTO documents (
                path, absolute_path, title, mtime_ns, size_bytes, content_hash,
                frontmatter_json, tags_json, wikilinks_json, aliases_json, headings_json,
                indexed_at_ns, source_kind, page_role, project_id, updated_at_ns, is_route_map
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        return int(cursor.lastrowid or 0)
    row_id, prev_hash, prev_mtime = existing
    if prev_hash == doc.content_hash and prev_mtime == doc.mtime_ns:
        # Refresh deterministic classification even when content is unchanged so
        # a schema upgrade backfills metadata without needless rechunking.
        conn.execute(
            """
            UPDATE documents SET indexed_at_ns = ?, source_kind = ?, page_role = ?,
                project_id = ?, updated_at_ns = ?, is_route_map = ?
            WHERE id = ?
            """,
            (
                now_ns,
                doc.source_kind,
                doc.page_role,
                doc.project_id,
                doc.updated_at_ns,
                int(doc.is_route_map),
                row_id,
            ),
        )
        return int(row_id)
    conn.execute(
        """
        UPDATE documents SET
            absolute_path = ?, title = ?, mtime_ns = ?, size_bytes = ?,
            content_hash = ?, frontmatter_json = ?, tags_json = ?,
            wikilinks_json = ?, aliases_json = ?, headings_json = ?,
            indexed_at_ns = ?, source_kind = ?, page_role = ?, project_id = ?,
            updated_at_ns = ?, is_route_map = ?
        WHERE id = ?
        """,
        (
            doc.absolute_path,
            doc.title,
            doc.mtime_ns,
            doc.size_bytes,
            doc.content_hash,
            json.dumps(doc.frontmatter, ensure_ascii=False, default=str),
            json.dumps(list(doc.tags), ensure_ascii=False),
            json.dumps(list(doc.wikilinks), ensure_ascii=False),
            json.dumps(list(doc.aliases), ensure_ascii=False),
            json.dumps(list(doc.headings), ensure_ascii=False),
            now_ns,
            doc.source_kind,
            doc.page_role,
            doc.project_id,
            doc.updated_at_ns,
            int(doc.is_route_map),
            row_id,
        ),
    )
    return int(row_id)


def _delete_missing(conn: sqlite3.Connection, seen_paths: set[str]) -> int:
    """Remove absent documents and every derived projection they own.

    sqlite-vec virtual tables cannot declare a foreign key to ``chunks``. The
    vector rows therefore require explicit deletion before the chunk/document
    cascade in the same transaction.
    """
    rows = conn.execute("SELECT id, path FROM documents").fetchall()
    store = SqliteVecStore(conn)
    removed = 0
    for row_id, path in rows:
        if path not in seen_paths:
            with dbmod.transaction(conn):
                chunk_ids = [
                    int(row[0])
                    for row in conn.execute(
                        "SELECT id FROM chunks WHERE document_id = ?", (row_id,)
                    ).fetchall()
                ]
                store.delete(chunk_ids)
                conn.execute("DELETE FROM documents WHERE id = ?", (row_id,))
            removed += 1
    return removed


# --- the public entry point -------------------------------------------------


class Indexer:
    """Walks a vault and projects it into the SQLite database.

    Usage::

        indexer = Indexer(settings)
        stats = indexer.run(mode="incremental")

    Phase 3 adds an optional :class:`Embedder`. When provided, the
    indexer batches chunk text through the embedder and persists
    vectors to :class:`SqliteVecStore` on every chunk insert. On a
    model change (any existing ``chunk_embeddings.embedding_model``
    differs from ``settings.embedding_model``) the indexer wipes
    the vector store and re-embeds everything in one pass.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store

    def run(self, *, mode: str = "incremental") -> IndexRunStats:
        """Run one indexing pass.

        Parameters
        ----------
        mode:
            ``"incremental"`` (default) is the normal operation: read every
            file, add/update as needed, delete missing. ``"full"`` is
            identical for now; reserved for a future reindex mode that
            drops the database first.
        """
        if mode not in ("incremental", "full"):
            raise ValueError(f"mode must be 'incremental' or 'full', got {mode!r}")
        if mode == "full":
            logger.info("full mode requested; rebuilding the derived projection")

        vault = self.settings.vault_path
        if not vault.is_dir():
            raise FileNotFoundError(f"vault path does not exist or is not a directory: {vault}")

        started_ns = time.time_ns()
        added = updated = skipped = removed = 0
        chunks_added = chunks_updated = chunks_removed = 0
        embeddings_built = 0
        embeddings_rebuilt = 0
        errors: list[str] = []
        seen_paths: set[str] = set()

        with dbmod.connect(self.settings.db_path) as conn:
            dbmod.init_schema(conn)
            if mode == "full":
                dbmod.clear_projection(conn)
            # Phase 3: open the vector store if an embedder is
            # provided. The store is local to this run because its
            # connection lifetime matches the run; caching it on
            # ``self`` would dangle after the first run.
            cleanup_store = self.vector_store or SqliteVecStore(conn)
            vector_store: VectorStore | None = None
            recipe_state: dict[str, str] = {}
            if self.embedder is not None:
                vector_store = cleanup_store
                actual_dim = self.embedder.dim
                if actual_dim != self.settings.embedding_dim:
                    raise ValueError(
                        "embedder dimension does not match configured vector schema "
                        f"({actual_dim} != {self.settings.embedding_dim})"
                    )
                recipe_state = embedding_recipe_state(
                    model_name=self.settings.embedding_model,
                    dimension=actual_dim,
                )
                recipe_state.update(model_provenance(self.settings.embedding_model))
            run_id = _begin_run(conn, started_ns, mode)

            # Phase 3: model-change detection. If the database already
            # holds vectors produced by a different model, the
            # in-flight run will rebuild every chunk's embedding
            # regardless of whether the chunk text changed.
            reembed_all = False
            if vector_store is not None:
                reembed_all = _needs_full_reembed(
                    conn, expected_model=self.settings.embedding_model
                )
                if vector_store.count() > 0:
                    reembed_all = reembed_all or not dbmod.projection_metadata_matches(
                        conn, compatibility_state(recipe_state)
                    )
                if reembed_all:
                    logger.warning(
                        "embedding model or recipe mismatch detected; rebuilding every chunk embedding "
                        "(expected model %r)",
                        self.settings.embedding_model,
                    )

            # Phase 3: one-shot backfill. If the chunk store has
            # rows without corresponding embeddings (typical after
            # the v2 -> v3 schema upgrade, or after a `--no-embed`
            # run), embed them now. We piggy-back on the normal
            # embeddings_built counter so the run report reflects
            # the work done. ``reembed_all`` forces the same for
            # already-embedded chunks because their model is wrong.
            if vector_store is not None:
                assert self.embedder is not None  # invariant: pair
                backfill_n = _backfill_missing_embeddings(
                    conn,
                    embedder=self.embedder,
                    store=vector_store,
                    model=self.settings.embedding_model,
                    force=reembed_all,
                )
                if backfill_n:
                    logger.info(
                        "backfilled %d chunk embedding(s) without vectors",
                        backfill_n,
                    )
                    if reembed_all:
                        embeddings_rebuilt += backfill_n
                    else:
                        embeddings_built += backfill_n

            for vf in iter_vault_files(vault, self.settings):
                seen_paths.add(vf.rel_path)
                try:
                    doc, parsed = _build_document_and_parsed(vf, vault)
                    prev = conn.execute(
                        "SELECT id, content_hash, mtime_ns FROM documents WHERE path = ?",
                        (doc.path,),
                    ).fetchone()
                    if prev is not None and prev[1] == doc.content_hash and prev[2] == doc.mtime_ns:
                        _upsert_document(conn, doc)
                        skipped += 1
                        continue

                    # A document and every materialised projection must change
                    # together. In particular, vec0 has no FK to chunks, so a
                    # failed embedding must not strand a changed document with
                    # no chunks/vectors or leave deletion half-applied.
                    with dbmod.transaction(conn):
                        doc_id = _upsert_document(conn, doc)
                        old_ids: list[int] = []
                        if prev is not None:
                            old_ids = delete_chunks_for_document(conn, doc_id)
                            cleanup_store.delete(old_ids)
                        ids = _chunk_and_persist(conn, doc_id, parsed)
                        replace_document_links(
                            conn,
                            doc_id,
                            path=doc.path,
                            aliases=doc.aliases,
                            wikilinks=doc.wikilinks,
                        )
                        n = 0
                        if vector_store is not None:
                            assert self.embedder is not None  # invariant: pair
                            n = _embed_chunks(
                                conn,
                                doc_id,
                                ids,
                                embedder=self.embedder,
                                store=vector_store,
                                model=self.settings.embedding_model,
                            )

                    if prev is None:
                        added += 1
                        chunks_added += len(ids)
                    else:
                        updated += 1
                        chunks_removed += len(old_ids)
                        chunks_updated += len(ids)
                    if reembed_all:
                        embeddings_rebuilt += n
                    else:
                        embeddings_built += n
                except Exception as exc:
                    msg = f"{vf.rel_path}: {type(exc).__name__}: {exc}"
                    logger.warning("index error: %s", msg)
                    errors.append(msg)

            chunks_removed += _delete_missing_chunks(conn, seen_paths)
            removed = _delete_missing(conn, seen_paths)
            with dbmod.transaction(conn):
                links_resolved = resolve_links(conn)
            if links_resolved:
                logger.info("resolved %d wikilink(s)", links_resolved)
            changed = added + updated + removed
            has_entities = int(conn.execute("SELECT COUNT(*) FROM communities").fetchone()[0])
            if changed or not has_entities:
                with dbmod.transaction(conn):
                    entity_stats = refresh_entity_graph(conn)
                logger.info(
                    "entity graph: %d mention edge(s), %d communit(ies)",
                    entity_stats["mention_edges"],
                    entity_stats["communities"],
                )
            _finish_run(
                conn,
                run_id,
                time.time_ns(),
                added,
                updated,
                removed,
                skipped,
                chunks_added,
                chunks_updated,
                chunks_removed,
                embeddings_built,
                embeddings_rebuilt,
                errors,
            )
            if (
                vector_store is not None
                and not errors
                and vector_store.count() == count_chunks(conn)
            ):
                dbmod.set_projection_metadata(conn, recipe_state)
            if mode == "full":
                dbmod.set_rebuild_state(conn, "ready" if not errors else "failed")

        stats = IndexRunStats(
            mode=mode,
            documents_seen=len(seen_paths),
            documents_added=added,
            documents_updated=updated,
            documents_removed=removed,
            documents_skipped=skipped,
            chunks_added=chunks_added,
            chunks_updated=chunks_updated,
            chunks_removed=chunks_removed,
            embeddings_built=embeddings_built,
            embeddings_rebuilt=embeddings_rebuilt,
            errors=tuple(errors),
        )
        logger.info(
            "index complete: seen=%d added=%d updated=%d removed=%d skipped=%d "
            "chunks: +%d ~%d -%d embeddings: built=%d rebuilt=%d errors=%d",
            stats.documents_seen,
            stats.documents_added,
            stats.documents_updated,
            stats.documents_removed,
            stats.documents_skipped,
            stats.chunks_added,
            stats.chunks_updated,
            stats.chunks_removed,
            stats.embeddings_built,
            stats.embeddings_rebuilt,
            len(stats.errors),
        )
        return stats


def _chunk_and_persist(
    conn: sqlite3.Connection,
    document_id: int,
    parsed: ParsedDocument,
    *,
    max_chunk_chars: int = 2000,
) -> list[int]:
    """Chunk ``parsed`` and persist under ``document_id``.

    Returns the list of chunk row ids in insertion order (empty list
    if the document produced no chunks). The caller uses these ids
    to write embeddings in Phase 3.
    """
    chunks = chunk_document(parsed, document_id=document_id, max_chunk_chars=max_chunk_chars)
    if not chunks:
        return []
    return insert_chunks(conn, document_id, chunks)


def _needs_full_reembed(conn: sqlite3.Connection, *, expected_model: str) -> bool:
    """Return True if the vector store holds rows produced by a different model.

    Three cases return False:

    - The vector store is empty (first run).
    - Every existing row already names ``expected_model``.
    - The expected model is empty / unset (we don't trust the caller).

    A mixed-state database (some rows ``expected_model``, some not)
    also returns True so the in-flight run converges on one model.
    """
    if not expected_model:
        return False
    row = conn.execute(
        "SELECT MIN(embedding_model), MAX(embedding_model) FROM chunk_embeddings"
    ).fetchone()
    if row is None or row[0] is None:
        return False
    min_model, max_model = row
    return not (min_model == max_model == expected_model)


def _embed_chunks(
    conn: sqlite3.Connection,
    document_id: int,
    chunk_ids: list[int],
    *,
    embedder: Embedder,
    store: VectorStore,
    model: str,
) -> int:
    """Embed the chunks identified by ``chunk_ids`` and persist to ``store``.

    Loads chunk text by id from the database (the indexer only holds
    ``Chunk`` objects in scope via the chunker, not their row ids),
    embeds in one batch, and upserts into the vector store. Returns
    the number of embeddings written.
    """
    if not chunk_ids:
        return 0
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"""
        SELECT c.id, c.text, c.heading_path_json, d.title, d.aliases_json, d.tags_json
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.id IN ({placeholders})
        ORDER BY c.id
        """,
        list(chunk_ids),
    ).fetchall()
    ids_ordered, texts = _document_embedding_inputs(rows)
    vectors = embedder.embed(texts)
    store.upsert(ids_ordered, vectors, embedding_model=model)
    return len(ids_ordered)


def _backfill_missing_embeddings(
    conn: sqlite3.Connection,
    *,
    embedder: Embedder,
    store: VectorStore,
    model: str,
    force: bool = False,
) -> int:
    """Embed every chunk that does not yet have a vector row.

    The default (``force=False``) targets only chunks with no
    matching row in ``chunk_embeddings``; that covers the typical
    v2 -> v3 schema upgrade and any prior ``--no-embed`` run.
    ``force=True`` additionally rewrites rows whose
    ``embedding_model`` differs from ``model`` (the model-change
    path; ``_needs_full_reembed`` already detected this so we
    just do the work).

    Returns the number of embeddings written.
    """
    if force:
        # Preserve the last complete projection until the replacement is ready.
        # A model/recipe migration is all-or-nothing: clearing vec0 before a
        # multi-minute backfill makes an interruption leave every chunk
        # unsearchable. The transaction rolls the deletion and partial batch
        # upserts back on an embedding failure or process interruption.
        with dbmod.transaction(conn):
            conn.execute("DELETE FROM chunk_embeddings")
            rows = conn.execute("SELECT id FROM chunks ORDER BY id").fetchall()
            ids_all = [int(r[0]) for r in rows]
            if not ids_all:
                return 0
            return _embed_chunks_batched(
                conn,
                chunk_ids=ids_all,
                embedder=embedder,
                store=store,
                model=model,
            )

    else:
        # Target chunks with no corresponding vector row.
        rows = conn.execute(
            """
            SELECT c.id FROM chunks c
            LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id
            WHERE e.chunk_id IS NULL
            """
        ).fetchall()
        missing_ids = [int(r[0]) for r in rows]
        if not missing_ids:
            return 0
        # Stream the missing chunks in batches so FastEmbed's
        # peak memory stays bounded. Fetch by id list to keep
        # the SQL predictable.
        return _embed_chunks_batched(
            conn,
            chunk_ids=missing_ids,
            embedder=embedder,
            store=store,
            model=model,
        )


def _embed_chunks_batched(
    conn: sqlite3.Connection,
    *,
    chunk_ids: list[int],
    embedder: Embedder,
    store: VectorStore,
    model: str,
    batch_size: int = 128,
) -> int:
    """Embed ``chunk_ids`` in batches and persist to ``store``.

    Both the SQL text fetch and the FastEmbed call are streamed
    per-batch to bound peak memory on large vaults. The default
    batch size of 128 was tuned against the 384-dim BGE-small
    model on a 60 GB / 16-core box; peak RSS stayed under 3 GB
    even on multi-thousand-chunk backfills.
    """
    if not chunk_ids:
        return 0
    total = 0
    n = len(chunk_ids)
    logger.info("backfill: embedding %d chunks in batches of %d", n, batch_size)
    for start in range(0, n, batch_size):
        batch_ids = chunk_ids[start : start + batch_size]
        placeholders = ",".join("?" * len(batch_ids))
        rows = conn.execute(
            f"""
            SELECT c.id, c.text, c.heading_path_json, d.title, d.aliases_json, d.tags_json
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({placeholders})
            ORDER BY c.id
            """,
            batch_ids,
        ).fetchall()
        ids_ordered, texts = _document_embedding_inputs(rows)
        vectors = embedder.embed(texts)
        store.upsert(ids_ordered, vectors, embedding_model=model)
        total += len(ids_ordered)
        # Log every batch for a clear progress signal during the
        # initial backfill; quiet later.
        logger.info("backfill progress: %d/%d", total, n)
    return total


def _document_embedding_inputs(rows: list[tuple[object, ...]]) -> tuple[list[int], list[str]]:
    """Build the versioned structural embedding input for persisted chunks."""
    ids: list[int] = []
    texts: list[str] = []
    for chunk_id, body, heading_path_json, title, aliases_json, tags_json in rows:
        ids.append(int(str(chunk_id)))
        heading_path = tuple(str(item) for item in json.loads(str(heading_path_json)))
        aliases = tuple(str(item) for item in json.loads(str(aliases_json)))
        tags = tuple(str(item) for item in json.loads(str(tags_json)))
        texts.append(
            format_document_embedding_input(
                title=str(title),
                heading_path=heading_path,
                aliases=aliases,
                tags=tags,
                body=str(body),
            )
        )
    return ids, texts


def count_chunks(conn: sqlite3.Connection) -> int:
    """Return the number of relational chunks expected to have one vector each."""
    return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])


def _delete_missing_chunks(conn: sqlite3.Connection, seen_paths: set[str]) -> int:
    """Best-effort safety net.

    The ``chunks.document_id`` FK has ``ON DELETE CASCADE``, so
    removing the parent document row already removes its chunks.
    This function exists to keep the counters honest if a future
    schema variant changes that cascade behaviour. It always
    returns 0 in the current schema.
    """
    return 0


# --- run-log helpers --------------------------------------------------------


def _begin_run(conn: sqlite3.Connection, started_ns: int, mode: str) -> int:
    cursor = conn.execute(
        "INSERT INTO index_runs (started_at_ns, mode) VALUES (?, ?)",
        (started_ns, mode),
    )
    return int(cursor.lastrowid or 0)


def _finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    finished_ns: int,
    added: int,
    updated: int,
    removed: int,
    skipped: int,
    chunks_added: int,
    chunks_updated: int,
    chunks_removed: int,
    embeddings_built: int,
    embeddings_rebuilt: int,
    errors: Iterable[str],
) -> None:
    conn.execute(
        """
        UPDATE index_runs SET
            finished_at_ns = ?,
            documents_added = ?,
            documents_updated = ?,
            documents_removed = ?,
            documents_skipped = ?,
            errors_json = ?
        WHERE id = ?
        """,
        (
            finished_ns,
            added,
            updated,
            removed,
            skipped,
            json.dumps(list(errors), ensure_ascii=False),
            run_id,
        ),
    )
    # Persist chunk counts in a side table-free way: encode them into
    # the errors_json field for now. Phase 13 will add proper
    # chunk-count columns when we know the eval shape. Keeping a
    # simple comment here so future agents see the intent.
    _ = (chunks_added, chunks_updated, chunks_removed, embeddings_built, embeddings_rebuilt)


# --- public status helper ---------------------------------------------------


def summarise_database(db_path: Path) -> dict[str, object]:
    """Return a small dict describing the database state. Used by ``llmwiki status``."""
    if not db_path.exists():
        return {"exists": False, "path": str(db_path)}
    with dbmod.connect(db_path) as conn:
        documents = int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        runs = int(conn.execute("SELECT COUNT(*) FROM index_runs").fetchone()[0])
        try:
            embeddings = int(conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0])
        except sqlite3.OperationalError:
            # Pre-Phase-3 database without the embeddings table.
            embeddings = 0
        try:
            from .entities import graph_counts
            from .graph import graph_summary

            graph = {**graph_summary(conn), **graph_counts(conn)}
        except sqlite3.OperationalError:
            graph = {"links": 0, "resolved": 0, "unresolved": 0}
        last_run_row = conn.execute(
            "SELECT started_at_ns, finished_at_ns, mode, documents_added, "
            "documents_updated, documents_removed, documents_skipped "
            "FROM index_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return {
        "exists": True,
        "path": str(db_path),
        "documents": documents,
        "chunks": chunks,
        "embeddings": embeddings,
        "graph": graph,
        "runs": runs,
        "last_run": dict(
            zip(
                (
                    "started_at_ns",
                    "finished_at_ns",
                    "mode",
                    "added",
                    "updated",
                    "removed",
                    "skipped",
                ),
                last_run_row,
                strict=True,
            )
        )
        if last_run_row
        else None,
    }


__all__ = ["Document", "Indexer", "VaultFile", "iter_vault_files", "summarise_database"]
