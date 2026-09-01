"""SQLite schema and connection helpers for the RAG core.

The database is a rebuildable projection of the canonical Markdown vault. It holds:

- Document metadata (path, mtime, content hash, frontmatter, tags, wikilinks, aliases).
- Chunk metadata (heading path, text, hash, position; Phase 2).
- Embeddings (Phase 3).
- Lexical index (Phase 4).
- Graph edges (Phase 10).
- Retrieval runs and eval results (Phase 13).

The schema is defined in code as an ordered migration registry. ``init_schema``
upgrades an empty v0 database and every released database through the
current version, committing each transition atomically.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from .logging import get_logger

logger = get_logger("db")

_SCHEMA_VERSION: Final = 7

# sqlite-vec needs the embedding dimension as a schema literal. The
# plan locks BGE-small-en-v1.5 (384-dim). If we ever swap models we
# drop and recreate the vec0 table via a future migration.
_EMBEDDING_DIM: Final = 384
_PRIVATE_DIRECTORY_MODE: Final = 0o700
_PRIVATE_FILE_MODE: Final = 0o600
_UMASK_LOCK = threading.Lock()

# Model name + dim are exported for the embedder / indexer.
EMBEDDING_DIM: Final = _EMBEDDING_DIM


def _projection_files(db_path: Path) -> tuple[Path, Path, Path]:
    """Return the main SQLite projection and its WAL sidecars."""
    return (
        db_path,
        db_path.with_name(f"{db_path.name}-wal"),
        db_path.with_name(f"{db_path.name}-shm"),
    )


def _secure_projection_storage(db_path: Path) -> None:
    """Create or repair private POSIX permissions for projection storage."""
    if os.name != "posix":
        logger.warning("projection permissions are not verified on this platform")
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(db_path.parent, _PRIVATE_DIRECTORY_MODE)
    fd = os.open(db_path, os.O_RDWR | os.O_CREAT, _PRIVATE_FILE_MODE)
    os.close(fd)
    for path in _projection_files(db_path):
        if path.exists():
            os.chmod(path, _PRIVATE_FILE_MODE)


@contextmanager
def _private_umask() -> Iterator[None]:
    """Keep SQLite-created WAL sidecars private while they are created."""
    if os.name != "posix":
        yield
        return
    with _UMASK_LOCK:
        previous = os.umask(0o077)
        try:
            yield
        finally:
            os.umask(previous)


def init_schema(conn: sqlite3.Connection) -> None:
    """Upgrade a database through each ordered, transactional migration."""
    previous_version = _schema_version(conn)
    if previous_version > _SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema version {previous_version} is newer than supported "
            f"version {_SCHEMA_VERSION}"
        )

    for version in range(previous_version, _SCHEMA_VERSION):
        # Do not use executescript here: it commits any open transaction before
        # running its SQL. Each migration's DDL and version marker must commit
        # or roll back together.
        with transaction(conn):
            _MIGRATIONS[version](conn)
            _set_schema_version(conn, version + 1)

    if previous_version < 2:
        _backfill_v2_chunks(conn)
    _backfill_orphan_chunks(conn)


def _schema_version(conn: sqlite3.Connection) -> int:
    """Return zero for an empty database and the stored version otherwise."""
    has_meta = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_meta'"
    ).fetchone()
    if has_meta is None:
        return 0
    row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        raise RuntimeError("database has schema_meta but no schema_version")
    return int(row[0])


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
        (str(version),),
    )


def _migrate_v0_to_v1(conn: sqlite3.Connection) -> None:
    """Create the original document and index-run projection tables."""
    for statement in (
        "CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        """CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL UNIQUE,
            absolute_path TEXT NOT NULL, title TEXT NOT NULL, mtime_ns INTEGER NOT NULL,
            size_bytes INTEGER NOT NULL, content_hash TEXT NOT NULL, frontmatter_json TEXT,
            tags_json TEXT NOT NULL DEFAULT '[]', wikilinks_json TEXT NOT NULL DEFAULT '[]',
            aliases_json TEXT NOT NULL DEFAULT '[]', headings_json TEXT NOT NULL DEFAULT '[]',
            indexed_at_ns INTEGER NOT NULL
        )""",
        "CREATE INDEX idx_documents_mtime ON documents(mtime_ns)",
        "CREATE INDEX idx_documents_hash ON documents(content_hash)",
        """CREATE TABLE index_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, started_at_ns INTEGER NOT NULL,
            finished_at_ns INTEGER, mode TEXT NOT NULL, documents_seen INTEGER NOT NULL DEFAULT 0,
            documents_added INTEGER NOT NULL DEFAULT 0, documents_updated INTEGER NOT NULL DEFAULT 0,
            documents_removed INTEGER NOT NULL DEFAULT 0, documents_skipped INTEGER NOT NULL DEFAULT 0,
            errors_json TEXT NOT NULL DEFAULT '[]'
        )""",
    ):
        conn.execute(statement)


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Add structural chunks to the v1 projection."""
    for statement in (
        """CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            position INTEGER NOT NULL, heading_path_json TEXT NOT NULL DEFAULT '[]',
            section_name TEXT NOT NULL DEFAULT '', text TEXT NOT NULL, text_hash TEXT NOT NULL,
            char_count INTEGER NOT NULL, indexed_at_ns INTEGER NOT NULL,
            UNIQUE (document_id, position)
        )""",
        "CREATE INDEX idx_chunks_document ON chunks(document_id)",
        "CREATE INDEX idx_chunks_hash ON chunks(text_hash)",
    ):
        conn.execute(statement)


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """Add sqlite-vec embeddings; ``connect`` loads sqlite-vec before this runs."""
    conn.execute(
        f"""CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding float[{_EMBEDDING_DIM}],
            embedding_model TEXT
        )"""
    )


def _migrate_v3_to_v4(conn: sqlite3.Connection) -> None:
    """Add durable projection metadata without rewriting indexed content."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS projection_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )"""
    )


def _migrate_v4_to_v5(conn: sqlite3.Connection) -> None:
    """Add path-derived corpus metadata required for profile-scoped retrieval."""
    existing_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(documents)")}
    columns = (
        ("source_kind", "TEXT NOT NULL DEFAULT 'operational'"),
        ("page_role", "TEXT NOT NULL DEFAULT 'operational'"),
        ("project_id", "TEXT"),
        ("updated_at_ns", "INTEGER NOT NULL DEFAULT 0"),
        ("is_route_map", "INTEGER NOT NULL DEFAULT 0"),
    )
    for name, definition in columns:
        if name not in existing_columns:
            conn.execute(f"ALTER TABLE documents ADD COLUMN {name} {definition}")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_corpus_profile "
        "ON documents(source_kind, page_role, project_id)"
    )


def _migrate_v5_to_v6(conn: sqlite3.Connection) -> None:
    """Add the trigger-maintained FTS5 lexical projection and backfill it.

    ``chunks_fts`` stores its own copy of the indexed columns (not an
    external-content table) so a cascading document delete can remove
    rows by rowid without needing the parent title, which is already
    gone by the time the chunk trigger fires.
    """
    from .lexical import FTS_TABLE, FTS_TOKENIZER, Fts5Index

    conn.execute(
        f"""CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} USING fts5(
            text, section_name, heading, title,
            tokenize = '{FTS_TOKENIZER}'
        )"""
    )
    conn.execute(
        f"""CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO {FTS_TABLE}(rowid, text, section_name, heading, title)
            VALUES (
                new.id, new.text, new.section_name,
                (SELECT group_concat(value, ' ') FROM json_each(new.heading_path_json)),
                (SELECT title FROM documents WHERE id = new.document_id)
            );
        END"""
    )
    conn.execute(
        f"""CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
            DELETE FROM {FTS_TABLE} WHERE rowid = old.id;
        END"""
    )
    Fts5Index(conn).rebuild()


def _migrate_v6_to_v7(conn: sqlite3.Connection) -> None:
    """Add the resolved wikilink graph projection (links + lookup keys)."""
    for statement in (
        """CREATE TABLE IF NOT EXISTS links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            target_document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
            target_text TEXT NOT NULL,
            target_key TEXT NOT NULL,
            target_path_hint TEXT
        )""",
        "CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_document_id)",
        "CREATE INDEX IF NOT EXISTS idx_links_target ON links(target_document_id)",
        "CREATE INDEX IF NOT EXISTS idx_links_key ON links(target_key)",
        """CREATE TABLE IF NOT EXISTS link_keys (
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            key TEXT NOT NULL,
            PRIMARY KEY (document_id, key)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_link_keys_key ON link_keys(key)",
    ):
        conn.execute(statement)
    # Backfill from persisted document metadata; resolution runs on the next index.
    from .graph import replace_document_links

    rows = conn.execute("SELECT id, path, aliases_json, wikilinks_json FROM documents").fetchall()
    import json as _json

    for doc_id, path, aliases_json, wikilinks_json in rows:
        replace_document_links(
            conn,
            int(doc_id),
            path=str(path),
            aliases=[str(a) for a in _json.loads(str(aliases_json or "[]"))],
            wikilinks=[str(w) for w in _json.loads(str(wikilinks_json or "[]"))],
        )


Migration = Callable[[sqlite3.Connection], None]
_MIGRATIONS: dict[int, Migration] = {
    0: _migrate_v0_to_v1,
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
    3: _migrate_v3_to_v4,
    4: _migrate_v4_to_v5,
    5: _migrate_v5_to_v6,
    6: _migrate_v6_to_v7,
}


def clear_projection(conn: sqlite3.Connection) -> None:
    """Clear projection rows and mark a full rebuild in progress.

    Reindexing can take many minutes. It is not held in one database
    transaction; the durable state marker prevents integrity/status surfaces
    from presenting an interrupted or partially rebuilt projection as healthy.
    """
    with transaction(conn):
        conn.execute("DELETE FROM chunk_embeddings")
        conn.execute("DELETE FROM chunks_fts")
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM links")
        conn.execute("DELETE FROM link_keys")
        conn.execute("DELETE FROM documents")
        conn.execute("DELETE FROM index_runs")
        conn.execute("DELETE FROM projection_meta")
        conn.execute(
            "INSERT INTO projection_meta(key, value) VALUES ('rebuild_state', 'in_progress')"
        )


def set_rebuild_state(conn: sqlite3.Connection, state: str) -> None:
    """Record the terminal state of a full projection rebuild."""
    if state not in {"ready", "failed"}:
        raise ValueError(f"unsupported rebuild state: {state!r}")
    with transaction(conn):
        conn.execute(
            "INSERT OR REPLACE INTO projection_meta(key, value) VALUES ('rebuild_state', ?)",
            (state,),
        )


def projection_metadata_matches(conn: sqlite3.Connection, expected: Mapping[str, str]) -> bool:
    """Return whether every expected compatibility key has the stored value."""
    if not expected:
        return True
    placeholders = ",".join("?" * len(expected))
    rows = conn.execute(
        f"SELECT key, value FROM projection_meta WHERE key IN ({placeholders})",
        list(expected),
    ).fetchall()
    return {str(key): str(value) for key, value in rows} == dict(expected)


def set_projection_metadata(conn: sqlite3.Connection, values: Mapping[str, str]) -> None:
    """Persist compatibility metadata atomically without disturbing rebuild state."""
    if not values:
        return
    with transaction(conn):
        conn.executemany(
            "INSERT OR REPLACE INTO projection_meta(key, value) VALUES (?, ?)",
            list(values.items()),
        )


def inspect_integrity(db_path: Path, *, vault_path: Path | None = None) -> dict[str, object]:
    """Report structural defects in the rebuildable retrieval projection.

    Missing embeddings are reported but not fatal because an explicit
    ``--no-embed`` index is supported. Orphan rows, vanished source documents,
    and mixed embedding models make the projection unsuitable for retrieval.
    """
    if not db_path.exists():
        return {"exists": False, "ok": False, "path": str(db_path)}
    with connect(db_path) as conn:
        try:
            schema_row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            schema_version = int(schema_row[0]) if schema_row is not None else None
        except sqlite3.OperationalError as exc:
            return {
                "exists": True,
                "ok": False,
                "path": str(db_path),
                "error": f"uninitialised or unsupported database: {exc}",
            }
        rebuild_state = "legacy"
        projection_meta_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'projection_meta'"
        ).fetchone()
        if projection_meta_exists is not None:
            state_row = conn.execute(
                "SELECT value FROM projection_meta WHERE key = 'rebuild_state'"
            ).fetchone()
            if state_row is not None:
                rebuild_state = str(state_row[0])
        orphan_vectors = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM chunk_embeddings e
                LEFT JOIN chunks c ON c.id = e.chunk_id
                WHERE c.id IS NULL
                """
            ).fetchone()[0]
        )
        orphan_chunks = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM chunks c
                LEFT JOIN documents d ON d.id = c.document_id
                WHERE d.id IS NULL
                """
            ).fetchone()[0]
        )
        chunks_without_embeddings = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM chunks c
                LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id
                WHERE e.chunk_id IS NULL
                """
            ).fetchone()[0]
        )
        fts_present = (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks_fts'"
            ).fetchone()
            is not None
        )
        orphan_fts_rows = 0
        chunks_without_fts = 0
        if fts_present:
            orphan_fts_rows = int(
                conn.execute(
                    "SELECT COUNT(*) FROM chunks_fts WHERE rowid NOT IN (SELECT id FROM chunks)"
                ).fetchone()[0]
            )
            chunks_without_fts = int(
                conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE id NOT IN (SELECT rowid FROM chunks_fts)"
                ).fetchone()[0]
            )
        links_present = (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'links'"
            ).fetchone()
            is not None
        )
        orphan_links = 0
        unresolved_links = 0
        if links_present:
            orphan_links = int(
                conn.execute(
                    "SELECT COUNT(*) FROM links WHERE source_document_id NOT IN (SELECT id FROM documents)"
                ).fetchone()[0]
            )
            unresolved_links = int(
                conn.execute(
                    "SELECT COUNT(*) FROM links WHERE target_document_id IS NULL"
                ).fetchone()[0]
            )
        models = [
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT embedding_model FROM chunk_embeddings ORDER BY embedding_model"
            ).fetchall()
        ]
        missing_on_disk: list[str] = []
        if vault_path is not None:
            for path, absolute_path in conn.execute(
                "SELECT path, absolute_path FROM documents ORDER BY path"
            ).fetchall():
                if not Path(str(absolute_path)).is_file():
                    missing_on_disk.append(str(path))

    ok = (
        schema_version == _SCHEMA_VERSION
        and not orphan_vectors
        and not orphan_chunks
        and not orphan_fts_rows
        and not chunks_without_fts
        and not orphan_links
        and not missing_on_disk
        and len(models) <= 1
        and rebuild_state not in {"in_progress", "failed"}
    )
    return {
        "exists": True,
        "ok": ok,
        "path": str(db_path),
        "schema_version": schema_version,
        "rebuild_state": rebuild_state,
        "orphan_vectors": orphan_vectors,
        "orphan_chunks": orphan_chunks,
        "chunks_without_embeddings": chunks_without_embeddings,
        "orphan_fts_rows": orphan_fts_rows,
        "chunks_without_fts": chunks_without_fts,
        "orphan_links": orphan_links,
        "unresolved_links": unresolved_links,
        "documents_missing_on_disk": missing_on_disk,
        "embedding_models": models,
        "mixed_embedding_models": len(models) > 1,
    }


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
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Make a group of projection writes atomic, nesting via savepoints."""
    if conn.in_transaction:
        conn.execute("SAVEPOINT llmwiki_projection")
        try:
            yield
        except Exception:
            conn.execute("ROLLBACK TO llmwiki_projection")
            conn.execute("RELEASE llmwiki_projection")
            raise
        else:
            conn.execute("RELEASE llmwiki_projection")
        return

    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection with the project's standard pragmas.

    Pragmas:
    - ``journal_mode=WAL``: concurrent readers, single writer.
    - ``foreign_keys=ON``: defensive; the schema doesn't yet use FKs
      but later phases (graph_edges) will.
    - ``synchronous=NORMAL``: WAL-safe; durable on commit.

    The ``sqlite-vec`` extension is loaded on every connection so
    the vec0 virtual table works. Loading is idempotent.

    The caller is responsible for committing; the context manager
    closes the connection on exit and rolls back on exception.
    """
    _secure_projection_storage(db_path)
    with _private_umask():
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.execute("PRAGMA journal_mode = WAL")
    try:
        _secure_projection_storage(db_path)
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        # Load sqlite-vec; required for the chunk_embeddings virtual table.
        import sqlite_vec  # local import to avoid hard dep at module import time

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


__all__ = [
    "clear_projection",
    "connect",
    "init_schema",
    "inspect_integrity",
    "set_rebuild_state",
    "transaction",
]
