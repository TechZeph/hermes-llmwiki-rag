"""Vector store interface and sqlite-vec implementation (Phase 3).

The :class:`VectorStore` ABC defines the contract every backend
satisfies. :class:`SqliteVecStore` is the only implementation we ship
in Phase 3 — it talks to the ``chunk_embeddings`` virtual table defined
in :mod:`llmwiki.db`.

Design notes:

- Similarity is **cosine**. sqlite-vec's ``vec_distance_cosine``
  returns the L2 distance between unit-normalised vectors (so for
  unit vectors at angle θ the distance is ``2 * sin(θ/2)`` — not
  ``1 - cos(θ)``). The returned ``score`` is the raw distance;
  smaller is more similar. The CLI's ``search`` command displays
  it as-is with a ``d=`` prefix so consumers know the metric.
- ``upsert`` is keyed on ``chunk_id`` (the integer PK of the chunks
  table) so the same store can serve chunk-level search and
  paragraph-level search without confusion.
- The ``embedding_model`` auxiliary column is set on every upsert. A
  future query can filter by model if the database ever holds
  mixed-model rows (e.g. during a model migration).
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Sequence


class VectorStore(ABC):
    """Abstract vector store.

    Implementations: :class:`SqliteVecStore` (Phase 3). Benchmarking
    may add a ``NumpyMemmapStore`` for sanity testing.
    """

    @abstractmethod
    def upsert(
        self,
        ids: Sequence[int],
        vectors: Sequence[Sequence[float]],
        *,
        embedding_model: str | None = None,
    ) -> None:
        """Insert or update vectors by row id. Idempotent.

        Both sequences must have the same length. ``vectors[i]`` is
        stored against ``ids[i]``. ``embedding_model`` is recorded
        as metadata so future runs can detect model changes; the
        base class does not require it.
        """

    @abstractmethod
    def search(self, query: Sequence[float], top_k: int) -> list[tuple[int, float]]:
        """Return ``(id, score)`` pairs ranked by similarity.

        Smaller ``score`` = more similar (it's the raw distance
        metric from the store). Callers that want a "higher is
        better" view should sort ascending. The store contract is
        intentionally unitless so different metrics can be swapped
        in without changing the indexer's code.
        """

    @abstractmethod
    def delete(self, ids: Sequence[int]) -> None:
        """Remove vectors by id. No-op for unknown ids."""

    @abstractmethod
    def count(self) -> int:
        """Number of vectors currently stored."""


class SqliteVecStore(VectorStore):
    """Vector store backed by a ``vec0`` virtual table in SQLite.

    The table must have columns ``chunk_id INTEGER PRIMARY KEY``,
    ``embedding float[N]``, and (optionally) ``embedding_model TEXT``.
    The dimension is fixed at schema time — see :mod:`llmwiki.db`.

    Parameters
    ----------
    conn:
        An open sqlite3 connection. The caller owns its lifetime; we
        do not close it.
    table:
        Name of the vec0 table. Defaults to ``chunk_embeddings``.
    """

    def __init__(self, conn: sqlite3.Connection, *, table: str = "chunk_embeddings") -> None:
        self._conn = conn
        self._table = table

    # --- write paths --------------------------------------------------------

    def upsert(
        self,
        ids: Sequence[int],
        vectors: Sequence[Sequence[float]],
        *,
        embedding_model: str | None = None,
    ) -> None:
        if len(ids) != len(vectors):
            raise ValueError(
                f"ids and vectors must have the same length "
                f"(got {len(ids)} ids, {len(vectors)} vectors)"
            )
        if not ids:
            return
        # sqlite-vec requires vectors to be bound as packed float32
        # bytes, not Python lists. The serializer is the canonical
        # helper from the package; using it keeps the byte layout
        # in lock-step with what MATCH will expect on the query side.
        import sqlite_vec  # local import to avoid hard dep at module import time

        rows = [
            (
                int(chunk_id),
                sqlite_vec.serialize_float32(list(vec)),
                # sqlite-vec's TEXT aux columns reject NULL, so use an
                # empty string when the caller doesn't supply a model.
                embedding_model if embedding_model is not None else "",
            )
            for chunk_id, vec in zip(ids, vectors, strict=True)
        ]
        with self._conn:
            # sqlite-vec virtual tables don't support the SQLite
            # ``ON CONFLICT ... DO UPDATE`` upsert syntax. We emulate
            # it with a DELETE for the same ids followed by INSERT.
            # Both run inside the same implicit transaction (the
            # ``with self._conn:`` block) so a crash mid-upsert can't
            # leave the store half-written.
            self._conn.executemany(
                f"DELETE FROM {self._table} WHERE chunk_id = ?",
                [(row[0],) for row in rows],
            )
            self._conn.executemany(
                f"""
                INSERT INTO {self._table}
                    (chunk_id, embedding, embedding_model)
                VALUES (?, ?, ?)
                """,
                rows,
            )

    def delete(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        with self._conn:
            self._conn.executemany(
                f"DELETE FROM {self._table} WHERE chunk_id = ?",
                [(int(i),) for i in ids],
            )

    def count(self) -> int:
        row = self._conn.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()
        return int(row[0]) if row else 0

    # --- read path ----------------------------------------------------------

    def search(self, query: Sequence[float], top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        import sqlite_vec  # local import to avoid hard dep at module import time

        q_bytes = sqlite_vec.serialize_float32([float(x) for x in query])
        # sqlite-vec KNN syntax: ``WHERE embedding MATCH ? ORDER BY
        # distance LIMIT ?``. ``distance`` is the L2 distance between
        # the (auto-normalised) vectors; smaller is more similar.
        # The VectorStore contract returns raw distance so the
        # metric is transparent to callers.
        rows = self._conn.execute(
            f"""
            SELECT chunk_id, distance
            FROM {self._table}
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (q_bytes, top_k),
        ).fetchall()
        return [(int(cid), float(dist)) for cid, dist in rows]


__all__ = ["SqliteVecStore", "VectorStore"]
