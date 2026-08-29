"""Unit tests for scoring helpers (Phase 6 contract)."""

from __future__ import annotations

from llmwiki.scoring import min_max_normalise


def test_normalises_to_unit_range() -> None:
    out = min_max_normalise([1.0, 2.0, 3.0, 4.0, 5.0])
    assert out == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_constant_input_returns_zeros() -> None:
    out = min_max_normalise([3.0, 3.0, 3.0])
    assert out == [0.0, 0.0, 0.0]


def test_empty_input_returns_empty() -> None:
    assert min_max_normalise([]) == []
