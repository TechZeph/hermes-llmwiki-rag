"""Tests for the vector store layer (Phase 3).

These are unit tests that do NOT require FastEmbed or any model
download. We use a fake ``Embedder`` subclass that returns
deterministic, hand-crafted vectors so the assertions are exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from llmwiki import db as dbmod
from llmwiki.embeddings import Embedder
from llmwiki.models import Chunk
from llmwiki.vector import SqliteVecStore


class FakeEmbedder(Embedder):
    """Deterministic embedder for tests.

    Produces ``dim``-dim vectors whose values are just the text
    repeated. This makes similarity assertions exact: identical
    texts produce identical vectors, distinct texts produce
    orthogonal vectors.

    Tests that go through the indexer must use ``dim=384`` because
    the production ``chunk_embeddings`` table is hardcoded to
    ``float[384]``. Pure-store tests can use any dim and pass a
    custom ``table`` to ``SqliteVecStore``.
    """

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "fake-embedder"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # One-hot at index derived from text length so identical
        # texts produce identical vectors.
        out: list[list[float]] = []
        for t in texts:
            idx = len(t) % self._dim
            v = [0.0] * self._dim
            v[idx] = 1.0
            out.append(v)
        return out


def _make_chunk(document_id: int, position: int, text: str) -> Chunk:
    return Chunk(
        id=None,
        document_id=document_id,
        heading_path=("T", f"S{position}"),
        section_name=f"S{position}",
        text=text,
        position=position,
    )


_TEST_DIM = 4


def _add_test_vec_table(conn, dim: int = _TEST_DIM) -> None:
    """Create a vec0 table sized for tests.

    The production schema hardcodes float[384] because that's the
    embedding model's dimension. For unit tests we need a smaller
    table; the schema-v3 database already exposes the production
    ``chunk_embeddings`` (float[384]) so we use a separate table
    name here. ``SqliteVecStore`` accepts a ``table`` argument.

    The test table mirrors the production schema shape (including
    the ``embedding_model`` aux column) so the same upsert path is
    exercised end-to-end.
    """
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS test_vectors USING vec0("
        f"chunk_id INTEGER PRIMARY KEY, embedding float[{dim}], "
        f"embedding_model TEXT)"
    )


# Helpers below are used by the indexer-level tests at the bottom
# of this file.


def _add_document(conn, path: str) -> int:
    cur = conn.execute(
        "INSERT INTO documents (path, absolute_path, title, mtime_ns, size_bytes, "
        "content_hash, indexed_at_ns) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (path, f"/{path}", path, 1, 1, "h", 1),
    )
    return int(cur.lastrowid or 0)


# All vector tests below use ``table="test_vectors"`` so they don't
# collide with the production 384-dim schema.


def _make_store(conn, dim: int = _TEST_DIM) -> SqliteVecStore:
    _add_test_vec_table(conn, dim=dim)
    return SqliteVecStore(conn, table="test_vectors")


def test_store_starts_empty(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    with dbmod.connect(db) as conn:
        dbmod.init_schema(conn)
        store = _make_store(conn)
        assert store.count() == 0
        assert store.search([0.0, 0.0, 0.0, 0.0], top_k=5) == []


def test_search_caps_requests_at_sqlite_vec_knn_limit(tmp_path: Path) -> None:
    """Search remains usable when callers request more than vec0 permits."""
    db = tmp_path / "test.sqlite"
    with dbmod.connect(db) as conn:
        dbmod.init_schema(conn)
        store = _make_store(conn)
        store.upsert([1], [[1.0, 0.0, 0.0, 0.0]])

        assert store.search([1.0, 0.0, 0.0, 0.0], top_k=4097) == [(1, 0.0)]


def test_upsert_and_search_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    with dbmod.connect(db) as conn:
        dbmod.init_schema(conn)
        store = _make_store(conn)
        # Hand-crafted unit vectors chosen so each pair has a
        # distinct distance with the query ``[1, 0, 0, 0]``:
        #   id 1: identical (d = 0)
        #   id 2: 60 deg   (d = 1)
        #   id 3: 90 deg   (d = sqrt(2) ≈ 1.414)
        store.upsert(
            [1, 2, 3],
            [[1, 0, 0, 0], [0.5, 0.866, 0, 0], [0, 0, 1, 0]],
        )
        assert store.count() == 3
        hits = store.search([1, 0, 0, 0], top_k=3)
        assert len(hits) == 3
        ids_in_order = [cid for cid, _ in hits]
        assert ids_in_order == [1, 2, 3]
        assert hits[0][1] == pytest.approx(0.0, abs=1e-6)
        assert hits[1][1] == pytest.approx(1.0, abs=1e-3)
        assert hits[2][1] == pytest.approx(2**0.5, abs=1e-3)


def test_upsert_overwrites_existing_row(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    with dbmod.connect(db) as conn:
        dbmod.init_schema(conn)
        store = _make_store(conn)
        store.upsert([1], [[1, 0, 0, 0]])
        store.upsert([1], [[0, 0, 0, 1]])
        assert store.count() == 1
        hits = store.search([0, 0, 0, 1], top_k=1)
        assert hits[0][0] == 1
        assert hits[0][1] == pytest.approx(0.0, abs=1e-6)


def test_upsert_records_embedding_model(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    with dbmod.connect(db) as conn:
        dbmod.init_schema(conn)
        # Add an embedding_model column for this assertion; not all
        # test tables carry it.
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS test_vectors_with_model USING vec0("
            "chunk_id INTEGER PRIMARY KEY, embedding float[4], embedding_model TEXT)"
        )
        store = SqliteVecStore(conn, table="test_vectors_with_model")
        store.upsert([1, 2], [[1, 0, 0, 0], [0, 1, 0, 0]], embedding_model="model-A")
        rows = conn.execute(
            "SELECT chunk_id, embedding_model FROM test_vectors_with_model ORDER BY chunk_id"
        ).fetchall()
        assert [(r[0], r[1]) for r in rows] == [(1, "model-A"), (2, "model-A")]


def test_delete_removes_rows(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    with dbmod.connect(db) as conn:
        dbmod.init_schema(conn)
        store = _make_store(conn)
        store.upsert([1, 2, 3], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        store.delete([2])
        assert store.count() == 2
        hits = store.search([0, 1, 0, 0], top_k=5)
        assert all(cid != 2 for cid, _ in hits)


def test_delete_unknown_ids_is_noop(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    with dbmod.connect(db) as conn:
        dbmod.init_schema(conn)
        store = _make_store(conn)
        store.upsert([1], [[1, 0, 0, 0]])
        store.delete([99, 100])
        assert store.count() == 1


def test_upsert_length_mismatch_raises(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    with dbmod.connect(db) as conn:
        dbmod.init_schema(conn)
        store = _make_store(conn)
        with pytest.raises(ValueError):
            store.upsert([1, 2], [[1, 0, 0, 0]])


def test_search_top_k_filters_results(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    with dbmod.connect(db) as conn:
        dbmod.init_schema(conn)
        store = _make_store(conn)
        store.upsert([1, 2, 3], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        assert len(store.search([1, 0, 0, 0], top_k=1)) == 1
        assert len(store.search([1, 0, 0, 0], top_k=2)) == 2


def test_search_returns_ascending_distance(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    with dbmod.connect(db) as conn:
        dbmod.init_schema(conn)
        store = _make_store(conn)
        store.upsert(
            [1, 2, 3],
            [[1, 0, 0, 0], [0.7071, 0.7071, 0, 0], [0, 0, 1, 0]],
        )
        hits = store.search([1, 0, 0, 0], top_k=3)
        distances = [d for _, d in hits]
        # Distances are non-decreasing (smaller = closer).
        assert distances == sorted(distances)
        assert hits[0][0] == 1  # closest to query


def test_indexer_writes_embeddings_for_added_documents(tmp_path: Path) -> None:
    """End-to-end: a single-doc vault produces both chunks and embeddings."""
    from llmwiki.config import Settings
    from llmwiki.indexer import Indexer

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text(
        "# Note\n\nFirst paragraph about apples.\n\nSecond paragraph about oranges.\n"
    )
    db_path = tmp_path / "test.sqlite"
    settings = Settings(
        vault_path=vault,
        db_path=db_path,
    )
    # Production schema is float[384]; FakeEmbedder must match.
    embedder = FakeEmbedder(dim=384)
    indexer = Indexer(settings, embedder=embedder)
    stats = indexer.run(mode="incremental")
    assert stats.documents_added == 1
    assert stats.chunks_added >= 1
    assert stats.embeddings_built == stats.chunks_added
    assert stats.errors == ()

    # The vector store should now contain embeddings for every chunk.
    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        store = SqliteVecStore(conn)
        n_chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        assert store.count() == n_chunks


def test_indexer_reembeds_on_model_change(tmp_path: Path) -> None:
    """A model mismatch forces every chunk to be re-embedded."""
    from llmwiki.config import Settings
    from llmwiki.indexer import Indexer

    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text("# Note\n\nbody text\n")
    db_path = tmp_path / "test.sqlite"
    settings = Settings(vault_path=vault, db_path=db_path)

    # First run: original model.
    indexer = Indexer(settings, embedder=FakeEmbedder(dim=384))
    s1 = indexer.run(mode="incremental")
    assert s1.embeddings_built >= 1

    # Manually rewrite the embedding_model column to simulate a
    # previous run under a different model.
    with dbmod.connect(db_path) as conn:
        conn.execute("UPDATE chunk_embeddings SET embedding_model = 'other-model'")

    # A model mismatch must rebuild every chunk even when source files did not change.
    indexer2 = Indexer(settings, embedder=FakeEmbedder(dim=384))
    s2 = indexer2.run(mode="incremental")
    assert s2.documents_updated == 0
    assert s2.documents_skipped == 1
    assert s2.embeddings_rebuilt >= 1
    # And nothing should be "built" because every chunk was already
    # in the database (no chunks added or updated by content change).
    assert s2.embeddings_built == 0


class FailingEmbedder(FakeEmbedder):
    """Fails only for the update fixture after its old projection exists."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if any("FAIL_EMBED" in text for text in texts):
            raise RuntimeError("injected embedding failure")
        return super().embed(texts)


class AlwaysFailingEmbedder(FakeEmbedder):
    """Fails a forced full re-embedding before it can replace old vectors."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("injected re-embedding failure")


def test_indexer_preserves_existing_vectors_when_full_reembedding_fails(tmp_path: Path) -> None:
    """A failed recipe/model migration never clears the last complete vector set."""
    from llmwiki.config import Settings
    from llmwiki.indexer import Indexer

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Note\n\noriginal body\n")
    db_path = tmp_path / "test.sqlite"
    settings = Settings(vault_path=vault, db_path=db_path)

    Indexer(settings, embedder=FakeEmbedder(dim=384)).run()
    with dbmod.connect(db_path) as conn:
        conn.execute("UPDATE chunk_embeddings SET embedding_model = 'legacy-model'")
        before_ids = [row[0] for row in conn.execute("SELECT chunk_id FROM chunk_embeddings")]

    with pytest.raises(RuntimeError, match="injected re-embedding failure"):
        Indexer(settings, embedder=AlwaysFailingEmbedder(dim=384)).run()

    with dbmod.connect(db_path) as conn:
        after_rows = conn.execute(
            "SELECT chunk_id, embedding_model FROM chunk_embeddings ORDER BY chunk_id"
        ).fetchall()
    assert [row[0] for row in after_rows] == before_ids
    assert {row[1] for row in after_rows} == {"legacy-model"}


def test_indexer_persists_fastembed_artifact_provenance_with_the_projection(tmp_path: Path) -> None:
    """The dense projection records its runtime package and model artifact source."""
    from llmwiki.config import Settings
    from llmwiki.indexer import Indexer

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Note\n\nbody\n", encoding="utf-8")
    db_path = tmp_path / "test.sqlite"

    Indexer(Settings(vault_path=vault, db_path=db_path), embedder=FakeEmbedder(dim=384)).run()

    with dbmod.connect(db_path) as conn:
        provenance = dict(
            conn.execute(
                "SELECT key, value FROM projection_meta "
                "WHERE key IN ('embedding.backend', 'embedding.backend_version', "
                "'embedding.artifact_source')"
            ).fetchall()
        )

    assert provenance["embedding.backend"] == "fastembed"
    assert provenance["embedding.backend_version"]
    assert provenance["embedding.artifact_source"] == "qdrant/bge-small-en-v1.5-onnx-q"


def test_indexer_rolls_back_document_projection_when_embedding_fails(tmp_path: Path) -> None:
    """A failed update leaves the prior document, chunks, and vectors intact."""
    from llmwiki.config import Settings
    from llmwiki.indexer import Indexer

    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text("# Note\n\noriginal body\n")
    db_path = tmp_path / "test.sqlite"
    settings = Settings(vault_path=vault, db_path=db_path)

    Indexer(settings, embedder=FailingEmbedder(dim=384)).run(mode="incremental")
    with dbmod.connect(db_path) as conn:
        before = conn.execute(
            "SELECT content_hash FROM documents WHERE path = 'note.md'"
        ).fetchone()[0]
        before_chunk_ids = [row[0] for row in conn.execute("SELECT id FROM chunks").fetchall()]
        before_vector_ids = [
            row[0] for row in conn.execute("SELECT chunk_id FROM chunk_embeddings").fetchall()
        ]

    note.write_text("# Note\n\nFAIL_EMBED changed body\n")
    stats = Indexer(settings, embedder=FailingEmbedder(dim=384)).run(mode="incremental")

    assert len(stats.errors) == 1
    with dbmod.connect(db_path) as conn:
        after = conn.execute(
            "SELECT content_hash FROM documents WHERE path = 'note.md'"
        ).fetchone()[0]
        after_chunk_ids = [row[0] for row in conn.execute("SELECT id FROM chunks").fetchall()]
        after_vector_ids = [
            row[0] for row in conn.execute("SELECT chunk_id FROM chunk_embeddings").fetchall()
        ]
    assert after == before
    assert after_chunk_ids == before_chunk_ids
    assert after_vector_ids == before_vector_ids


def test_indexer_deleting_document_removes_its_vectors(tmp_path: Path) -> None:
    """A deleted source removes every derived row, including vec0 rows."""
    from llmwiki.config import Settings
    from llmwiki.indexer import Indexer

    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text("# Note\n\nbody text\n")
    db_path = tmp_path / "test.sqlite"
    settings = Settings(vault_path=vault, db_path=db_path)

    Indexer(settings, embedder=FakeEmbedder(dim=384)).run(mode="incremental")
    note.unlink()

    stats = Indexer(settings, embedder=FakeEmbedder(dim=384)).run(mode="incremental")
    assert stats.documents_removed == 1
    with dbmod.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0] == 0


def test_indexer_updates_embeddings_when_chunks_change(tmp_path: Path) -> None:
    """A document edit drops the old vectors and writes new ones."""
    from llmwiki.config import Settings
    from llmwiki.indexer import Indexer

    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text("# Note\n\noriginal content\n")
    db_path = tmp_path / "test.sqlite"
    settings = Settings(vault_path=vault, db_path=db_path)

    indexer = Indexer(settings, embedder=FakeEmbedder(dim=384))
    s1 = indexer.run(mode="incremental")
    assert s1.embeddings_built >= 1

    # Modify the document.
    note.write_text("# Note\n\nupdated content with more text\n")
    # Force a fresh mtime so the indexer's mtime-based change
    # detection notices.
    import os

    stat = note.stat()
    os.utime(note, ns=(stat.st_atime_ns + 1, stat.st_mtime_ns + 1))

    s2 = indexer.run(mode="incremental")
    assert s2.documents_updated == 1
    assert s2.embeddings_built >= 1
    assert s2.errors == ()
    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        rows = conn.execute("SELECT embedding_model FROM chunk_embeddings").fetchall()
        # Every row must be tagged with the configured model name.
        assert all(r[0] == settings.embedding_model for r in rows)


def test_indexer_no_embed_mode_skips_embeddings(tmp_path: Path) -> None:
    """Without an embedder, chunks are still written but no vectors."""
    from llmwiki.config import Settings
    from llmwiki.indexer import Indexer

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Note\n\nbody\n")
    db_path = tmp_path / "test.sqlite"
    settings = Settings(vault_path=vault, db_path=db_path)

    indexer = Indexer(settings)  # no embedder
    s = indexer.run(mode="incremental")
    assert s.embeddings_built == 0
    assert s.embeddings_rebuilt == 0
    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        store = SqliteVecStore(conn)
        assert store.count() == 0


def test_indexer_backfills_chunks_without_vectors(tmp_path: Path) -> None:
    """A run after `--no-embed` (or after a v2->v3 schema upgrade)
    backfills embeddings for chunks that have no vector row."""
    from llmwiki.config import Settings
    from llmwiki.indexer import Indexer

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nalpha content\n")
    (vault / "b.md").write_text("# B\n\nbeta content\n")
    db_path = tmp_path / "test.sqlite"
    settings = Settings(vault_path=vault, db_path=db_path)

    # First run with no embedder: chunks persist, vectors don't.
    Indexer(settings).run(mode="incremental")
    with dbmod.connect(db_path) as conn:
        n_chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        assert n_chunks >= 2

    # Second run WITH an embedder: backfill should kick in.
    s = Indexer(settings, embedder=FakeEmbedder(dim=384)).run(mode="incremental")
    assert s.embeddings_built >= 2
    assert s.errors == ()
    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        store = SqliteVecStore(conn)
        # Every chunk now has a vector.
        n_chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        assert store.count() == n_chunks


def test_indexer_force_reembed_rebuilds_every_chunk(tmp_path: Path) -> None:
    """When the configured embedding model changes, every chunk is rebuilt."""
    from llmwiki.config import Settings
    from llmwiki.indexer import Indexer

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\ncontent\n")
    db_path = tmp_path / "test.sqlite"
    settings = Settings(vault_path=vault, db_path=db_path)

    # First run with the production model.
    s1 = Indexer(settings, embedder=FakeEmbedder(dim=384)).run(mode="incremental")
    assert s1.embeddings_built >= 1

    # Manually relabel every embedding row to a different model.
    with dbmod.connect(db_path) as conn:
        conn.execute("UPDATE chunk_embeddings SET embedding_model = 'old-model'")

    # Second run with no file changes: model mismatch triggers
    # ``force=True`` backfill, which wipes and rebuilds.
    s2 = Indexer(settings, embedder=FakeEmbedder(dim=384)).run(mode="incremental")
    assert s2.embeddings_rebuilt >= 1
    with dbmod.connect(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT embedding_model FROM chunk_embeddings").fetchall()
        assert all(r[0] == settings.embedding_model for r in rows)
