"""Unit tests for reciprocal-rank fusion and document diversification."""

from __future__ import annotations

import pytest

from llmwiki.hybrid import diversify, reciprocal_rank_fusion


def test_single_channel_preserves_order() -> None:
    fused = reciprocal_rank_fusion({"dense": [1, 2, 3]})
    assert [e.id for e in fused] == [1, 2, 3]
    assert fused[0].ranks == {"dense": 1}
    assert fused[0].rrf_score == pytest.approx(1 / 61)


def test_ids_in_both_channels_outrank_single_channel_ids() -> None:
    fused = reciprocal_rank_fusion({"dense": [1, 2, 3], "lexical": [2, 1, 4]})
    assert {e.id for e in fused[:2]} == {1, 2}
    assert fused[0].ranks == {"dense": 1, "lexical": 2}


def test_ties_break_by_best_rank_then_id() -> None:
    fused = reciprocal_rank_fusion({"dense": [5], "lexical": [3]})
    assert [e.id for e in fused] == [3, 5]


def test_duplicate_id_within_one_channel_counts_once() -> None:
    fused = reciprocal_rank_fusion({"dense": [1, 1, 2]})
    assert [e.id for e in fused] == [1, 2]
    assert fused[0].rrf_score == pytest.approx(1 / 61)


def test_rrf_k_changes_relative_weight_of_ranks() -> None:
    low_k = reciprocal_rank_fusion({"a": [1, 2], "b": [2]}, k=0)
    high_k = reciprocal_rank_fusion({"a": [1, 2], "b": [2]}, k=1000)
    assert [e.id for e in low_k] == [1, 2] or [e.id for e in low_k] == [2, 1]
    assert [e.id for e in high_k] == [2, 1]


def test_negative_k_rejected() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"a": [1]}, k=-1)


def test_diversify_caps_per_group_and_keeps_order() -> None:
    groups = {1: 10, 2: 10, 3: 10, 4: 20, 5: 10}
    assert diversify([1, 2, 3, 4, 5], groups, max_per_group=2) == [1, 2, 4]
    assert diversify([1, 2, 3, 4, 5], groups, max_per_group=0) == [1, 2, 3, 4, 5]
