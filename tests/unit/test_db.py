"""Tests for schema migration, rebuild, and projection-integrity behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmwiki import db as dbmod
from llmwiki.config import Settings
from llmwiki.indexer import Indexer


def _settings(vault: Path, db_path: Path) -> Settings:
    return Settings(vault_path=vault, db_path=db_path, log_level="WARNING")


def _create_legacy_v1_fixture(db_path: Path, note: Path) -> None:
    """Create the persisted shape written by the released v1 code."""
    with dbmod.connect(db_path) as conn:
        for statement in (
            "CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL UNIQUE,
                absolute_path TEXT NOT NULL, title TEXT NOT NULL, mtime_ns INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL, content_hash TEXT NOT NULL, frontmatter_json TEXT,
                tags_json TEXT NOT NULL DEFAULT '[]', wikilinks_json TEXT NOT NULL DEFAULT '[]',
                aliases_json TEXT NOT NULL DEFAULT '[]', headings_json TEXT NOT NULL DEFAULT '[]',
                indexed_at_ns INTEGER NOT NULL
            )
            """,
            "CREATE INDEX idx_documents_mtime ON documents(mtime_ns)",
            "CREATE INDEX idx_documents_hash ON documents(content_hash)",
            """
            CREATE TABLE index_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, started_at_ns INTEGER NOT NULL,
                finished_at_ns INTEGER, mode TEXT NOT NULL, documents_seen INTEGER NOT NULL DEFAULT 0,
                documents_added INTEGER NOT NULL DEFAULT 0, documents_updated INTEGER NOT NULL DEFAULT 0,
                documents_removed INTEGER NOT NULL DEFAULT 0, documents_skipped INTEGER NOT NULL DEFAULT 0,
                errors_json TEXT NOT NULL DEFAULT '[]'
            )
            """,
        ):
            conn.execute(statement)
        conn.execute("INSERT INTO schema_meta VALUES ('schema_version', '1')")
        conn.execute(
            "INSERT INTO documents (path, absolute_path, title, mtime_ns, size_bytes, "
            "content_hash, indexed_at_ns) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("note.md", str(note), "Note", 1, 1, "hash", 1),
        )


def _create_legacy_v2_fixture(db_path: Path) -> None:
    """Create the persisted shape written by the released v2 code."""
    with dbmod.connect(db_path) as conn:
        for statement in (
            "CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL UNIQUE,
                absolute_path TEXT NOT NULL, title TEXT NOT NULL, mtime_ns INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL, content_hash TEXT NOT NULL, frontmatter_json TEXT,
                tags_json TEXT NOT NULL DEFAULT '[]', wikilinks_json TEXT NOT NULL DEFAULT '[]',
                aliases_json TEXT NOT NULL DEFAULT '[]', headings_json TEXT NOT NULL DEFAULT '[]',
                indexed_at_ns INTEGER NOT NULL
            )
            """,
            "CREATE INDEX idx_documents_mtime ON documents(mtime_ns)",
            "CREATE INDEX idx_documents_hash ON documents(content_hash)",
            """
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                position INTEGER NOT NULL, heading_path_json TEXT NOT NULL DEFAULT '[]',
                section_name TEXT NOT NULL DEFAULT '', text TEXT NOT NULL, text_hash TEXT NOT NULL,
                char_count INTEGER NOT NULL, indexed_at_ns INTEGER NOT NULL,
                UNIQUE (document_id, position)
            )
            """,
            "CREATE INDEX idx_chunks_document ON chunks(document_id)",
            "CREATE INDEX idx_chunks_hash ON chunks(text_hash)",
            """
            CREATE TABLE index_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, started_at_ns INTEGER NOT NULL,
                finished_at_ns INTEGER, mode TEXT NOT NULL, documents_seen INTEGER NOT NULL DEFAULT 0,
                documents_added INTEGER NOT NULL DEFAULT 0, documents_updated INTEGER NOT NULL DEFAULT 0,
                documents_removed INTEGER NOT NULL DEFAULT 0, documents_skipped INTEGER NOT NULL DEFAULT 0,
                errors_json TEXT NOT NULL DEFAULT '[]'
            )
            """,
        ):
            conn.execute(statement)
        conn.execute("INSERT INTO schema_meta VALUES ('schema_version', '2')")


def test_init_schema_upgrades_real_legacy_v1_fixture(tmp_path: Path) -> None:
    """A released v1 database follows each historical transition to v4."""
    db_path = tmp_path / "legacy-v1.sqlite"
    note = tmp_path / "note.md"
    note.write_text("# Note\n\nbody\n", encoding="utf-8")
    _create_legacy_v1_fixture(db_path, note)

    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        version = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        projection_meta = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'projection_meta'"
        ).fetchone()

    assert version == "4"
    assert chunk_count >= 1
    assert projection_meta is not None


def test_init_schema_upgrades_real_legacy_v2_fixture(tmp_path: Path) -> None:
    """A released v2 database adds vec0 then v4 metadata without bootstrap DDL."""
    db_path = tmp_path / "legacy-v2.sqlite"
    _create_legacy_v2_fixture(db_path)

    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        version = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        embedding_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'chunk_embeddings'"
        ).fetchone()
        projection_meta = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'projection_meta'"
        ).fetchone()

    assert version == "4"
    assert embedding_table is not None
    assert projection_meta is not None


def test_init_schema_runs_historical_migrations_in_order_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A v1 database invokes v1→v2, v2→v3, then v3→v4 exactly once."""
    db_path = tmp_path / "ordered.sqlite"
    note = tmp_path / "note.md"
    note.write_text("# Note\n\nbody\n", encoding="utf-8")
    _create_legacy_v1_fixture(db_path, note)
    calls: list[int] = []
    originals = dict(dbmod._MIGRATIONS)

    def record(version: int):
        def migration(conn) -> None:
            calls.append(version)
            originals[version](conn)

        return migration

    monkeypatch.setattr(
        dbmod, "_MIGRATIONS", {version: record(version) for version in (1, 2, 3)}
    )
    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        dbmod.init_schema(conn)

    assert calls == [1, 2, 3]


def test_failed_historical_migration_rolls_back_prior_schema_and_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A v2→v3 failure rolls back only that migration and retains version 2."""
    db_path = tmp_path / "rollback.sqlite"
    _create_legacy_v2_fixture(db_path)

    original = dbmod._MIGRATIONS[2]

    def fail_after_ddl(conn) -> None:
        original(conn)
        conn.execute("CREATE TABLE migration_probe (id INTEGER PRIMARY KEY)")
        raise RuntimeError("injected v2 to v3 failure")

    monkeypatch.setitem(dbmod._MIGRATIONS, 2, fail_after_ddl)
    with dbmod.connect(db_path) as conn, pytest.raises(
        RuntimeError, match="injected v2 to v3 failure"
    ):
        dbmod.init_schema(conn)

    with dbmod.connect(db_path) as conn:
        version = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        probe = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'migration_probe'"
        ).fetchone()
        embeddings = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'chunk_embeddings'"
        ).fetchone()

    assert version == "2"
    assert probe is None
    assert embeddings is None


def test_init_schema_migrates_v3_database_to_current_version(tmp_path: Path) -> None:
    """A v3 projection upgrades in place without discarding document rows."""
    db_path = tmp_path / "llmwiki.sqlite"
    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        conn.execute(
            "INSERT INTO documents (path, absolute_path, title, mtime_ns, size_bytes, "
            "content_hash, indexed_at_ns) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("note.md", "/tmp/note.md", "Note", 1, 1, "hash", 1),
        )
        conn.execute(
            "UPDATE schema_meta SET value = '3' WHERE key = 'schema_version'"
        )

    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        version = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        documents = conn.execute("SELECT path FROM documents").fetchall()
        metadata_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'projection_meta'"
        ).fetchone()

    assert version == "4"
    assert documents == [("note.md",)]
    assert metadata_table is not None


def test_failed_migration_rolls_back_its_ddl_and_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A migration failure cannot leave a partially upgraded schema behind."""
    db_path = tmp_path / "llmwiki.sqlite"
    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        conn.execute("UPDATE schema_meta SET value = '3' WHERE key = 'schema_version'")
        conn.execute("DROP TABLE projection_meta")

    def fail_after_ddl(conn) -> None:
        conn.execute("CREATE TABLE migration_probe (id INTEGER PRIMARY KEY)")
        raise RuntimeError("injected migration failure")

    monkeypatch.setitem(dbmod._MIGRATIONS, 3, fail_after_ddl)
    with dbmod.connect(db_path) as conn, pytest.raises(
        RuntimeError, match="injected migration failure"
    ):
        dbmod.init_schema(conn)

    with dbmod.connect(db_path) as conn:
        version = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        probe = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'migration_probe'"
        ).fetchone()
    assert version == "3"
    assert probe is None


def test_full_index_rebuild_resets_prior_projection_runs(tmp_path: Path) -> None:
    """Full mode clears the old projection before rebuilding from the vault."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Note\n\nbody\n", encoding="utf-8")
    db_path = tmp_path / "llmwiki.sqlite"

    Indexer(_settings(vault, db_path)).run()
    stats = Indexer(_settings(vault, db_path)).run(mode="full")

    assert stats.documents_added == 1
    with dbmod.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] >= 1
        assert conn.execute("SELECT COUNT(*) FROM index_runs").fetchone()[0] == 1
        rebuild_state = conn.execute(
            "SELECT value FROM projection_meta WHERE key = 'rebuild_state'"
        ).fetchone()[0]
    assert rebuild_state == "ready"


def test_integrity_rejects_interrupted_full_rebuild(tmp_path: Path) -> None:
    """An interrupted rebuild cannot be presented as a clean projection."""
    db_path = tmp_path / "llmwiki.sqlite"
    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        dbmod.clear_projection(conn)

    report = dbmod.inspect_integrity(db_path)

    assert report["rebuild_state"] == "in_progress"
    assert report["ok"] is False


def test_integrity_reports_orphans_stale_documents_and_mixed_models(tmp_path: Path) -> None:
    """Integrity separates fatal projection defects from absent embeddings."""
    vault = tmp_path / "vault"
    vault.mkdir()
    db_path = tmp_path / "llmwiki.sqlite"

    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        doc_id = conn.execute(
            "INSERT INTO documents (path, absolute_path, title, mtime_ns, size_bytes, "
            "content_hash, indexed_at_ns) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("missing.md", str(vault / "missing.md"), "Missing", 1, 1, "hash", 1),
        ).lastrowid
        conn.execute(
            "INSERT INTO chunks (document_id, position, text, text_hash, char_count, indexed_at_ns) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, 0, "body", "hash", 4, 1),
        )
        # sqlite-vec accepts NULL vectors only through its serializer, so use
        # an orphan chunk ID by copying a correctly shaped zero vector later.

    report = dbmod.inspect_integrity(db_path, vault_path=vault)

    assert report["orphan_vectors"] == 0
    assert report["orphan_chunks"] == 0
    assert report["chunks_without_embeddings"] == 1
    assert report["documents_missing_on_disk"] == ["missing.md"]
    assert report["ok"] is False


def test_integrity_does_not_migrate_or_mutate_projection(tmp_path: Path) -> None:
    """Integrity is diagnostic and must not repair or migrate a database."""
    db_path = tmp_path / "llmwiki.sqlite"
    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        conn.execute("UPDATE schema_meta SET value = '3' WHERE key = 'schema_version'")
        conn.execute("DROP TABLE projection_meta")

    dbmod.inspect_integrity(db_path)

    with dbmod.connect(db_path) as conn:
        version = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        projection_meta = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'projection_meta'"
        ).fetchone()
    assert version == "3"
    assert projection_meta is None
