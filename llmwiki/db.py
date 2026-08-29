"""SQLite schema and connection helpers for the RAG core.

The database is the single source of truth for:

- Document metadata (path, mtime, content hash, frontmatter, tags, wikilinks, aliases).
- Chunk metadata (heading path, text, hash, position; Phase 2).
- Embeddings (Phase 3).
- Lexical index (Phase 4).
- Graph edges (Phase 10).
- Retrieval runs and eval results (Phase 13).

The schema is defined in code (not as a migration tool) for the
first 4-5 phases. When the schema starts changing, we'll introduce
``migrations/`` and a tiny migrator. For now, ``init_schema`` is
idempotent: it creates the tables if they don't exist.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from .logging import get_logger

logger = get_logger("db")

_SCHEMA_VERSION: Final = 2

_SCHEMA_SQL: Final = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    path            TEXT    NOT NULL UNIQUE,         -- relative to vault root, forward-slashes
    absolute_path   TEXT    NOT NULL,                -- for debugging / external tools
    title           TEXT    NOT NULL,                -- from H1 or frontmatter
    mtime_ns        INTEGER NOT NULL,                -- POSIX nanoseconds
    size_bytes      INTEGER NOT NULL,
    content_hash    TEXT    NOT NULL,                -- sha256 of the raw bytes
    frontmatter_json TEXT,                           -- JSON; NULL if absent
    tags_json       TEXT    NOT NULL DEFAULT '[]',   -- JSON list of strings
    wikilinks_json  TEXT    NOT NULL DEFAULT '[]',   -- JSON list of strings (raw targets)
    aliases_json    TEXT    NOT NULL DEFAULT '[]',   -- JSON list of strings
    headings_json   TEXT    NOT NULL DEFAULT '[]',   -- JSON list of {level, text} dicts
    indexed_at_ns   INTEGER NOT NULL                 -- POSIX nanoseconds of the last successful index
);

CREATE INDEX IF NOT EXISTS idx_documents_mtime ON documents(mtime_ns);
CREATE INDEX IF NOT EXISTS idx_documents_hash  ON documents(content_hash);

-- Phase 2: structural chunks. One row per (document, position) pair.
-- ``heading_path_json`` is a JSON list of strings (the breadcrumb);
-- ``section_name`` is the last element (the most specific heading).
CREATE TABLE IF NOT EXISTS chunks (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position          INTEGER NOT NULL,             -- ordinal within the document
    heading_path_json TEXT    NOT NULL DEFAULT '[]',
    section_name      TEXT    NOT NULL DEFAULT '',
    text              TEXT    NOT NULL,
    text_hash         TEXT    NOT NULL,              -- sha256 of the chunk text (for fast equality)
    char_count        INTEGER NOT NULL,
    indexed_at_ns     INTEGER NOT NULL,
    UNIQUE (document_id, position)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash     ON chunks(text_hash);

-- Phase 3 will add: chunk_embeddings (sqlite-vec)
-- Phase 4 will add: chunks_fts (FTS5 virtual table)
-- Phase 10 will add: graph_edges
-- Phase 13 will add: retrieval_runs, eval_results

CREATE TABLE IF NOT EXISTS index_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at_ns   INTEGER NOT NULL,
    finished_at_ns  INTEGER,
    mode            TEXT    NOT NULL,                 -- "full" | "incremental"
    documents_seen  INTEGER NOT NULL DEFAULT 0,
    documents_added INTEGER NOT NULL DEFAULT 0,
    documents_updated INTEGER NOT NULL DEFAULT 0,
    documents_removed INTEGER NOT NULL DEFAULT 0,
    documents_skipped INTEGER NOT NULL DEFAULT 0,
    errors_json     TEXT    NOT NULL DEFAULT '[]'
);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables and indexes if they do not yet exist.

    Also records the current schema version. Safe to call repeatedly.

    Schema migrations:

    - v1 → v2: introduced the ``chunks`` table. Existing v1 databases
      had document rows but no chunks. ``_backfill_v2_chunks`` is
      called once to populate chunks for every existing document.
      The backfill is also a no-op safety net on every run: if any
      document is missing chunks, they are filled in. This catches
      a backfill that crashed mid-run, and any future schema
      migration that adds a new derived table.
    """
    conn.executescript(_SCHEMA_SQL)
    previous_version_row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    previous_version = int(previous_version_row[0]) if previous_version_row else 0
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
        ("schema_version", str(_SCHEMA_VERSION)),
    )
    if previous_version < 2:
        _backfill_v2_chunks(conn)
    # Always run the orphan-chunks backfill: cheap (single query),
    # and it makes the system self-healing for any document that
    # lost its chunks (e.g. due to a crash mid-write).
    _backfill_orphan_chunks(conn)
    conn.commit()


def _backfill_v2_chunks(conn: sqlite3.Connection) -> None:
    """Populate chunks for every document that has none.

    Kept for the v1 → v2 migration path. The unconditional
    ``_backfill_orphan_chunks`` is what runs on every subsequent
    index; this function exists only so the v1 → v2 jump logs a
    one-time message.
    """
    _backfill_orphan_chunks(conn, log=True)


def _backfill_orphan_chunks(conn: sqlite3.Connection, *, log: bool = False) -> None:
    """Populate chunks for every document that has none.

    Runs on every ``init_schema`` call. Cheap (one SELECT), and it
    makes the system self-healing: any document whose chunks went
    missing (crash mid-write, manual SQL, etc.) is re-chunked on
    the next index.
    """
    from .indexer import _build_document_and_parsed, _chunk_and_persist

    rows = conn.execute(
        "SELECT d.id, d.absolute_path FROM documents d "
        "WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)"
    ).fetchall()
    if not rows:
        return
    if log:
        logger.info("schema v1->v2 backfill: chunking %d existing documents", len(rows))
    else:
        logger.info("orphan-chunks backfill: chunking %d documents", len(rows))
    from pathlib import Path

    from .indexer import VaultFile

    for doc_id, abs_path in rows:
        try:
            vf_path = Path(abs_path)
            if not vf_path.exists():
                continue
            vf = VaultFile(
                rel_path="",  # unused by the chunker
                abs_path=vf_path,
                mtime_ns=vf_path.stat().st_mtime_ns,
                size_bytes=vf_path.stat().st_size,
            )
            _doc, parsed = _build_document_and_parsed(vf, vf_path.parent)
            _chunk_and_persist(conn, int(doc_id), parsed)
        except Exception as exc:
            logger.warning("orphan-chunks backfill failed for %s: %s", abs_path, exc)


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection with the project's standard pragmas.

    Pragmas:
    - ``journal_mode=WAL``: concurrent readers, single writer.
    - ``foreign_keys=ON``: defensive; the schema doesn't yet use FKs
      but later phases (graph_edges) will.
    - ``synchronous=NORMAL``: WAL-safe; durable on commit.

    The caller is responsible for committing; the context manager
    closes the connection on exit and rolls back on exception.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


__all__ = ["connect", "init_schema"]
