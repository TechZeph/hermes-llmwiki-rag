"""Deterministic decomposition and multi-query fusion."""

from __future__ import annotations

import pytest

from llmwiki.models import Candidate, RetrievalResult
from llmwiki.multiquery import decompose_query, fuse_results, retrieve_multiquery


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("what is sqlite-vec?", ["what is sqlite-vec?"]),
        ("short and sweet", ["short and sweet"]),
        (
            "what is reciprocal rank fusion and how does the hermes rag project use it?",
            ["what is reciprocal rank fusion", "how does the hermes rag project use it"],
        ),
        (
            "compare the urban farm success criteria vs the home par levels ones",
            ["compare the urban farm success criteria", "the home par levels ones"],
        ),
        (
            "which reranker models were considered; what did gate R decide?",
            ["which reranker models were considered", "what did gate R decide"],
        ),
        (
            "bread and butter pudding recipe from the notes",
            ["bread and butter pudding recipe from the notes"],
        ),
    ],
)
def test_decompose_query(query: str, expected: list[str]) -> None:
    assert decompose_query(query) == expected


def _cand(i: int, path: str) -> Candidate:
    return Candidate(
        chunk_id=i,
        document_id=i,
        path=path,
        title=path,
        heading_path=(path,),
        section_name="",
        position=0,
        text="t",
        text_hash="h",
        source_kind="wiki",
        page_role="durable",
        project_id=None,
        updated_at_ns=1,
        is_route_map=False,
        rrf_score=0.1,
        selection_reason="hybrid",
    )


def _res(*cands: Candidate) -> RetrievalResult:
    return RetrievalResult(
        query="q",
        profile="answer",
        mode="hybrid",
        candidates=tuple(cands),
        intent="general",
        elapsed_ms=5.0,
    )


def test_fuse_results_rewards_pages_hit_by_several_parts() -> None:
    a, b, c = _cand(1, "wiki/a.md"), _cand(2, "wiki/b.md"), _cand(3, "wiki/c.md")
    fused = fuse_results("q", ["p1", "p2"], [_res(a, c), _res(b, c)], top_k=10)
    assert fused.candidates[0].chunk_id == 3  # c appears in both
    assert fused.mode == "multiquery" and fused.elapsed_ms == 10.0
    assert fused.candidates[0].selection_reason.startswith("multiquery[q1,q2]")


def test_retrieve_multiquery_falls_back_for_single_clause() -> None:
    calls: list[str] = []

    def fake(q: str) -> RetrievalResult:
        calls.append(q)
        return _res(_cand(1, "wiki/a.md"))

    retrieve_multiquery("what is sqlite-vec?", fake, top_k=5)
    assert calls == ["what is sqlite-vec?"]
    calls.clear()
    retrieve_multiquery(
        "what is reciprocal rank fusion and how does the hermes rag project use it?", fake, top_k=5
    )
    assert len(calls) == 3 and calls[-1].startswith("what is reciprocal rank fusion and")
