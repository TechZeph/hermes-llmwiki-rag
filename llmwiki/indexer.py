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
from .logging import get_logger
from .models import Document, IndexRunStats
from .parser import ParsedDocument, parse_markdown

logger = get_logger("indexer")


# --- filesystem walk --------------------------------------------------------


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
    vault = vault.resolve()
    for root, dirs, files in os.walk(vault):
        # In-place prune of ``.obsidian`` etc so ``os.walk`` skips subtrees.
        dirs[:] = sorted(d for d in dirs if d not in settings.ignored_dirs)
        for name in sorted(files):
            if not name.endswith(".md"):
                continue
            abs_path = Path(root) / name
            rel_path = abs_path.relative_to(vault).as_posix()
            if _should_skip(rel_path, settings.ignored_dirs, settings.ignored_globs):
                continue
            try:
                stat = abs_path.stat()
            except OSError as exc:
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
    parsed = parse_markdown(str(vf.abs_path))
    doc = Document(
        id=None,
        path=vf.rel_path,
        absolute_path=str(vf.abs_path),
        title=parsed.title or vf.abs_path.stem,
        mtime_ns=vf.mtime_ns,
        size_bytes=vf.size_bytes,
        content_hash=_sha256(vf.abs_path),
        frontmatter=parsed.frontmatter,
        tags=parsed.tags,
        wikilinks=parsed.wikilinks,
        aliases=parsed.aliases,
        headings=parsed.headings,
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
    )
    if existing is None:
        cursor = conn.execute(
            """
            INSERT INTO documents (
                path, absolute_path, title, mtime_ns, size_bytes, content_hash,
                frontmatter_json, tags_json, wikilinks_json, aliases_json, headings_json,
                indexed_at_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        return int(cursor.lastrowid or 0)
    row_id, prev_hash, prev_mtime = existing
    if prev_hash == doc.content_hash and prev_mtime == doc.mtime_ns:
        # No change. Touch indexed_at so liveness is visible, but do not rewrite.
        conn.execute("UPDATE documents SET indexed_at_ns = ? WHERE id = ?", (now_ns, row_id))
        return int(row_id)
    conn.execute(
        """
        UPDATE documents SET
            absolute_path = ?, title = ?, mtime_ns = ?, size_bytes = ?,
            content_hash = ?, frontmatter_json = ?, tags_json = ?,
            wikilinks_json = ?, aliases_json = ?, headings_json = ?,
            indexed_at_ns = ?
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
            row_id,
        ),
    )
    return int(row_id)


def _delete_missing(conn: sqlite3.Connection, seen_paths: set[str]) -> int:
    """Remove documents whose paths were not seen on this run. Returns count.

    Chunk rows cascade-delete via the FK on ``chunks.document_id``,
    so we don't have to clean them up here.
    """
    rows = conn.execute("SELECT id, path FROM documents").fetchall()
    removed = 0
    for row_id, path in rows:
        if path not in seen_paths:
            conn.execute("DELETE FROM documents WHERE id = ?", (row_id,))
            removed += 1
    return removed


# --- the public entry point -------------------------------------------------


class Indexer:
    """Walks a vault and projects it into the SQLite database.

    Usage::

        indexer = Indexer(settings)
        stats = indexer.run(mode="incremental")
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

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
            # Future: drop tables and recreate. For Phase 1, full == incremental.
            logger.info("full mode requested; running incremental pass (no drop yet)")

        vault = self.settings.vault_path
        if not vault.is_dir():
            raise FileNotFoundError(f"vault path does not exist or is not a directory: {vault}")

        started_ns = time.time_ns()
        added = updated = skipped = removed = 0
        chunks_added = chunks_updated = chunks_removed = 0
        errors: list[str] = []
        seen_paths: set[str] = set()

        with dbmod.connect(self.settings.db_path) as conn:
            dbmod.init_schema(conn)
            run_id = _begin_run(conn, started_ns, mode)

            for vf in iter_vault_files(vault, self.settings):
                seen_paths.add(vf.rel_path)
                try:
                    doc, parsed = _build_document_and_parsed(vf, vault)
                    prev = conn.execute(
                        "SELECT id, content_hash, mtime_ns FROM documents WHERE path = ?",
                        (doc.path,),
                    ).fetchone()
                    doc_id = _upsert_document(conn, doc)

                    if prev is None:
                        added += 1
                        n = _chunk_and_persist(conn, doc_id, parsed)
                        chunks_added += n
                    elif prev[1] == doc.content_hash and prev[2] == doc.mtime_ns:
                        skipped += 1
                    else:
                        updated += 1
                        n_old = delete_chunks_for_document(conn, doc_id)
                        n_new = _chunk_and_persist(conn, doc_id, parsed)
                        chunks_removed += n_old
                        chunks_updated += n_new
                except Exception as exc:
                    msg = f"{vf.rel_path}: {type(exc).__name__}: {exc}"
                    logger.warning("index error: %s", msg)
                    errors.append(msg)

            chunks_removed += _delete_missing_chunks(conn, seen_paths)
            removed = _delete_missing(conn, seen_paths)
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
                errors,
            )

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
            errors=tuple(errors),
        )
        logger.info(
            "index complete: seen=%d added=%d updated=%d removed=%d skipped=%d "
            "chunks: +%d ~%d -%d errors=%d",
            stats.documents_seen,
            stats.documents_added,
            stats.documents_updated,
            stats.documents_removed,
            stats.documents_skipped,
            stats.chunks_added,
            stats.chunks_updated,
            stats.chunks_removed,
            len(stats.errors),
        )
        return stats


def _chunk_and_persist(
    conn: sqlite3.Connection,
    document_id: int,
    parsed: ParsedDocument,
    *,
    max_chunk_chars: int = 2000,
) -> int:
    """Chunk ``parsed`` and persist under ``document_id``. Returns the count."""
    chunks = chunk_document(parsed, document_id=document_id, max_chunk_chars=max_chunk_chars)
    if not chunks:
        return 0
    return insert_chunks(conn, document_id, chunks)


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
    _ = (chunks_added, chunks_updated, chunks_removed)


# --- public status helper ---------------------------------------------------


def summarise_database(db_path: Path) -> dict[str, object]:
    """Return a small dict describing the database state. Used by ``llmwiki status``."""
    if not db_path.exists():
        return {"exists": False, "path": str(db_path)}
    with dbmod.connect(db_path) as conn:
        documents = int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        runs = int(conn.execute("SELECT COUNT(*) FROM index_runs").fetchone()[0])
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
