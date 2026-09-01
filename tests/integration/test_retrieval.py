"""End-to-end retrieval over a synthetic vault: dense, lexical, hybrid."""

from __future__ import annotations

import pytest

from llmwiki import db as dbmod
from llmwiki.config import Settings
from llmwiki.indexer import Indexer
from llmwiki.reranker import Reranker
from llmwiki.retrieval import Retriever
from tests.helpers import SAMPLE_KEYWORDS, SAMPLE_VAULT, KeywordEmbedder, write_vault


@pytest.fixture(scope="module")
def indexed(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("vault")
    vault = write_vault(root / "v", SAMPLE_VAULT)
    settings = Settings(vault_path=vault, db_path=root / "db.sqlite")
    embedder = KeywordEmbedder(SAMPLE_KEYWORDS)
    stats = Indexer(settings, embedder=embedder).run()
    assert not stats.errors
    return settings, embedder


@pytest.fixture
def retriever(indexed):
    settings, embedder = indexed
    with dbmod.connect(settings.db_path) as conn:
        dbmod.init_schema(conn)
        yield Retriever(conn, embedder=embedder, settings=settings)


def _paths(result) -> list[str]:
    return [c.path for c in result.candidates]


def test_dense_mode_uses_only_vectors(retriever) -> None:
    result = retriever.retrieve("sqlite-vec", mode="dense", profile="all", max_per_document=0)
    assert result.lexical_returned == 0
    assert result.dense_returned > 0
    assert all(c.dense_rank is not None and c.lexical_rank is None for c in result.candidates)
    assert result.candidates[0].dense_distance == pytest.approx(0.0)


def test_lexical_mode_uses_only_fts(retriever) -> None:
    result = retriever.retrieve("Condorcet", mode="lexical", profile="all")
    assert result.dense_returned == 0
    assert _paths(result) == ["raw/papers/rrf-paper.md"]
    assert result.candidates[0].bm25_score is not None and result.candidates[0].bm25_score > 0
    assert result.candidates[0].dense_rank is None


def test_hybrid_fuses_both_channels_and_keeps_raw_metrics(retriever) -> None:
    # "arena" is only a dense keyword hit; "Condorcet" only a lexical hit.
    result = retriever.retrieve("arena Condorcet", mode="hybrid", profile="all")
    paths = _paths(result)
    assert "wiki/fastembed.md" in paths
    assert "raw/papers/rrf-paper.md" in paths
    fast = next(c for c in result.candidates if c.path == "wiki/fastembed.md")
    assert fast.rrf_score is not None
    assert fast.dense_rank is not None or fast.lexical_rank is not None
    assert result.fused_total >= 2


def test_default_answer_profile_excludes_raw_log_route_maps_and_root_files(retriever) -> None:
    result = retriever.retrieve(
        "sqlite-vec embeddings", mode="hybrid", top_k=50, max_per_document=0
    )
    paths = set(_paths(result))
    assert paths
    assert all(p.startswith("wiki/") for p in paths)
    assert "wiki/log.md" not in paths
    assert "wiki/index-tools.md" not in paths
    assert "wiki/projects/rag/index.md" not in paths
    assert not any(p.startswith(("raw/", "Clippings/")) for p in paths)


def test_explicit_profiles_scope_the_corpus(retriever) -> None:
    ev = retriever.retrieve("sqlite-vec embeddings", mode="hybrid", profile="evidence", top_k=20)
    assert set(_paths(ev)) <= {"raw/papers/rrf-paper.md", "Clippings/ideas/vector-idea.md"}
    assert {c.authority_class for c in ev.candidates} <= {"evidence", "idea"}
    hist = retriever.retrieve("sqlite-vec", mode="hybrid", profile="history")
    assert _paths(hist) == ["wiki/log.md"]
    proj = retriever.retrieve(
        "sqlite-vec", mode="hybrid", profile="project:rag", top_k=20, max_per_document=0
    )
    assert set(_paths(proj)) == {
        "wiki/projects/rag/current-state.md",
        "wiki/projects/rag/decisions.md",
    }


def test_authority_promotes_current_state_for_status_questions(retriever) -> None:
    result = retriever.retrieve("what is the current status of sqlite-vec?", mode="hybrid", top_k=5)
    assert result.intent == "current-state"
    assert result.candidates[0].path == "wiki/projects/rag/current-state.md"
    assert result.candidates[0].authority_match is True
    decision = retriever.retrieve(
        "why did we choose sqlite-vec instead of faiss?", mode="hybrid", top_k=5
    )
    assert decision.intent == "decision"
    assert decision.candidates[0].path == "wiki/projects/rag/decisions.md"


def test_diversification_caps_chunks_per_document(retriever) -> None:
    capped = retriever.retrieve(
        "sqlite-vec", mode="hybrid", profile="all", top_k=20, max_per_document=1
    )
    assert len({c.document_id for c in capped.candidates}) == len(capped.candidates)
    uncapped = retriever.retrieve(
        "sqlite-vec", mode="hybrid", profile="all", top_k=20, max_per_document=0
    )
    assert len(uncapped.candidates) >= len(capped.candidates)


def test_empty_query_and_bad_mode(retriever) -> None:
    assert retriever.retrieve("   ", mode="hybrid").candidates == ()
    with pytest.raises(ValueError):
        retriever.retrieve("x", mode="magic")
    with pytest.raises(ValueError):
        retriever.retrieve("x", profile="nope")


class ReverseReranker(Reranker):
    @property
    def model_name(self) -> str:
        return "reverse"

    def rerank(self, query, documents, *, top_k=None):
        return [(i, float(i)) for i in range(len(documents))][::-1]


def test_reranker_reorders_head_and_records_scores(indexed) -> None:
    settings, embedder = indexed
    with dbmod.connect(settings.db_path) as conn:
        r = Retriever(conn, embedder=embedder, settings=settings, reranker=ReverseReranker())
        common = {"mode": "hybrid", "profile": "all", "top_k": 100, "max_per_document": 0}
        plain = r.retrieve("sqlite-vec", rerank=False, apply_authority=False, **common)
        reranked = r.retrieve("sqlite-vec", rerank=True, apply_authority=False, **common)
    assert len(plain.candidates) <= settings.rerank_candidates
    assert [c.chunk_id for c in reranked.candidates] == [c.chunk_id for c in plain.candidates][::-1]
    assert all(c.rerank_score is not None for c in reranked.candidates)
    assert all("rerank" in c.selection_reason for c in reranked.candidates)


def test_context_block_from_real_retrieval(indexed) -> None:
    from llmwiki.citations import ENVELOPE_CLOSE, ENVELOPE_OPEN
    from llmwiki.retrieval import context_for

    settings, embedder = indexed
    with dbmod.connect(settings.db_path) as conn:
        r = Retriever(conn, embedder=embedder, settings=settings)
        result = r.retrieve("what is the current status of sqlite-vec?", mode="hybrid", top_k=6)
        block = context_for(result, settings)
    assert not block.empty
    assert block.text.startswith(ENVELOPE_OPEN) and block.text.rstrip().endswith(ENVELOPE_CLOSE)
    assert block.citations[0].path == "wiki/projects/rag/current-state.md"
    assert block.citations[0].authority_class == "current-state"
    assert block.total_tokens <= settings.context_budget_tokens
    assert all(not c.path.startswith("/") for c in block.citations)
    assert all(c.chunk_ids and c.content_hashes for c in block.citations)
