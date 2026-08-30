"""Tests for the chunk persistence layer (Phase 2)."""

from __future__ import annotations

from pathlib import Path

from llmwiki import db as dbmod
from llmwiki.chunks import (
    count_chunks,
    delete_chunks_for_document,
    insert_chunks,
    iter_chunks_for_document,
)
from llmwiki.models import Chunk


def _make_chunk(document_id: int, position: int, text: str) -> Chunk:
    return Chunk(
        id=None,
        document_id=document_id,
        heading_path=("Title", f"S{position}"),
        section_name=f"S{position}",
        text=text,
        position=position,
    )


def test_insert_chunks_persists_rows(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    with dbmod.connect(db) as conn:
        dbmod.init_schema(conn)
        cur = conn.execute(
            "INSERT INTO documents (path, absolute_path, title, mtime_ns, size_bytes, "
            "content_hash, indexed_at_ns) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("a.md", "/a.md", "A", 1, 1, "h", 1),
        )
        doc_id = int(cur.lastrowid or 0)
        insert_chunks(
            conn,
            doc_id,
            [_make_chunk(doc_id, 0, "first"), _make_chunk(doc_id, 1, "second")],
        )
        assert count_chunks(conn) == 2


def test_insert_chunks_replaces_existing(tmp_path: Path) -> None:
    """Calling insert_chunks twice replaces the old rows; count stays at the new size."""
    db = tmp_path / "test.sqlite"
    with dbmod.connect(db) as conn:
        dbmod.init_schema(conn)
        cur = conn.execute(
            "INSERT INTO documents (path, absolute_path, title, mtime_ns, size_bytes, "
            "content_hash, indexed_at_ns) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("a.md", "/a.md", "A", 1, 1, "h", 1),
        )
        doc_id = int(cur.lastrowid or 0)
        insert_chunks(
            conn, doc_id, [_make_chunk(doc_id, 0, "v1-a"), _make_chunk(doc_id, 1, "v1-b")]
        )
        # Now overwrite with a different number of chunks.
        insert_chunks(conn, doc_id, [_make_chunk(doc_id, 0, "v2-a")])
        rows = conn.execute(
            "SELECT text FROM chunks WHERE document_id = ? ORDER BY position", (doc_id,)
        ).fetchall()
        assert [r[0] for r in rows] == ["v2-a"]


def test_delete_chunks_for_document_removes_all_rows(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    with dbmod.connect(db) as conn:
        dbmod.init_schema(conn)
        cur = conn.execute(
            "INSERT INTO documents (path, absolute_path, title, mtime_ns, size_bytes, "
            "content_hash, indexed_at_ns) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("a.md", "/a.md", "A", 1, 1, "h", 1),
        )
        doc_id = int(cur.lastrowid or 0)
        insert_chunks(conn, doc_id, [_make_chunk(doc_id, 0, "x"), _make_chunk(doc_id, 1, "y")])
        assert count_chunks(conn) == 2
        removed_ids = delete_chunks_for_document(conn, doc_id)
        assert len(removed_ids) == 2
        assert count_chunks(conn) == 0


def test_iter_chunks_returns_in_position_order(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    with dbmod.connect(db) as conn:
        dbmod.init_schema(conn)
        cur = conn.execute(
            "INSERT INTO documents (path, absolute_path, title, mtime_ns, size_bytes, "
            "content_hash, indexed_at_ns) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("a.md", "/a.md", "A", 1, 1, "h", 1),
        )
        doc_id = int(cur.lastrowid or 0)
        # Insert out of order; iter_chunks should reorder.
        insert_chunks(
            conn,
            doc_id,
            [
                _make_chunk(doc_id, 2, "third"),
                _make_chunk(doc_id, 0, "first"),
                _make_chunk(doc_id, 1, "second"),
            ],
        )
        chunks = iter_chunks_for_document(conn, doc_id)
        assert [c.text for c in chunks] == ["first", "second", "third"]
        assert [c.position for c in chunks] == [0, 1, 2]
        # All have ids assigned.
        assert all(c.id is not None for c in chunks)


def test_deleting_document_cascades_to_chunks(tmp_path: Path) -> None:
    """The FK has ON DELETE CASCADE; verify it works through SQLite."""
    db = tmp_path / "test.sqlite"
    with dbmod.connect(db) as conn:
        dbmod.init_schema(conn)
        cur = conn.execute(
            "INSERT INTO documents (path, absolute_path, title, mtime_ns, size_bytes, "
            "content_hash, indexed_at_ns) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("a.md", "/a.md", "A", 1, 1, "h", 1),
        )
        doc_id = int(cur.lastrowid or 0)
        insert_chunks(conn, doc_id, [_make_chunk(doc_id, 0, "x")])
        assert count_chunks(conn) == 1
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        assert count_chunks(conn) == 0
