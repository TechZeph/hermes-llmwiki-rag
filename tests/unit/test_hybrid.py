"""Unit tests for the hybrid retrieval fusion helper (Phase 5 contract)."""

from __future__ import annotations

from llmwiki.hybrid import reciprocal_rank_fusion


def test_single_list_returns_same_ordering() -> None:
    fused = reciprocal_rank_fusion([("a", 1.0), ("b", 0.5), ("c", 0.1)])
    assert [doc_id for doc_id, _ in fused] == ["a", "b", "c"]


def test_two_lists_merge_with_rrf() -> None:
    dense = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
    lexical = [("b", 0.95), ("a", 0.6), ("d", 0.4)]
    fused = reciprocal_rank_fusion(dense, lexical)
    # Both 'a' and 'b' appear in both lists, so they should outrank 'c' and 'd'.
    top_two = {doc_id for doc_id, _ in fused[:2]}
    assert top_two == {"a", "b"}


def test_disjoint_lists_concatenate() -> None:
    fused = reciprocal_rank_fusion([("a", 1.0)], [("b", 1.0)])
    assert {doc_id for doc_id, _ in fused} == {"a", "b"}
