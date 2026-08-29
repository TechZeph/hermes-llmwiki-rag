"""Score normalisation and combination utilities (Phase 6+ placeholder)."""

from __future__ import annotations

from collections.abc import Iterable


def min_max_normalise(scores: Iterable[float]) -> list[float]:
    """Linearly normalise a list of scores to ``[0, 1]``.

    If all scores are equal, returns a list of zeros. An empty
    input returns an empty list.
    """
    s = list(scores)
    if not s:
        return []
    lo, hi = min(s), max(s)
    if hi == lo:
        return [0.0 for _ in s]
    return [(x - lo) / (hi - lo) for x in s]


__all__ = ["min_max_normalise"]
