"""Chunk persistence helpers (Phase 2).

The indexer is the only caller of these functions in Phase 2. They
are kept in their own module so that Phase 3 (embeddings) and
Phase 4 (FTS5) can reuse the read paths without dragging in the
indexer.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterable

from .models import Chunk

__all__ = [
    "count_chunks",
    "delete_chunks_for_document",
    "insert_chunks",
    "iter_chunks_for_document",
]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def insert_chunks(conn: sqlite3.Connection, document_id: int, chunks: Iterable[Chunk]) -> int:
    """Replace all chunks for ``document_id`` with the given list.

    Strategy: delete-then-insert in a single transaction. This is
    the right call for an incremental indexer because the chunk set
    for a changed document is fully derived from the document body
    and the chunker is deterministic — there's no value in
    diffing old vs. new.

    Returns the number of chunks inserted.
    """
    chunks_list = list(chunks)
    now_ns = time.time_ns()
    with conn:
        delete_chunks_for_document(conn, document_id)
        if not chunks_list:
            return 0
        rows = [
            (
                document_id,
                chunk.position,
                json.dumps(list(chunk.heading_path), ensure_ascii=False),
                chunk.section_name,
                chunk.text,
                _hash_text(chunk.text),
                len(chunk.text),
                now_ns,
            )
            for chunk in chunks_list
        ]
        conn.executemany(
            """
            INSERT INTO chunks (
                document_id, position, heading_path_json, section_name,
                text, text_hash, char_count, indexed_at_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(chunks_list)


def delete_chunks_for_document(conn: sqlite3.Connection, document_id: int) -> int:
    """Remove all chunks for a document. Returns the rowcount."""
    cursor = conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    return int(cursor.rowcount or 0)


def iter_chunks_for_document(conn: sqlite3.Connection, document_id: int) -> list[Chunk]:
    """Return all chunks for a document in position order."""
    rows = conn.execute(
        """
        SELECT id, document_id, position, heading_path_json, section_name, text
        FROM chunks WHERE document_id = ? ORDER BY position
        """,
        (document_id,),
    ).fetchall()
    return [
        Chunk(
            id=int(row[0]),
            document_id=int(row[1]),
            position=int(row[2]),
            heading_path=tuple(json.loads(row[3])),
            section_name=str(row[4]),
            text=str(row[5]),
        )
        for row in rows
    ]


def count_chunks(conn: sqlite3.Connection) -> int:
    """Return the total number of chunks across all documents."""
    row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    return int(row[0]) if row else 0
