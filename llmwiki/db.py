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

_SCHEMA_VERSION: Final = 1

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

-- Phase 2 will add: chunks, chunks_fts, chunk_embeddings
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
    """
    conn.executescript(_SCHEMA_SQL)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
        ("schema_version", str(_SCHEMA_VERSION)),
    )
    conn.commit()


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
