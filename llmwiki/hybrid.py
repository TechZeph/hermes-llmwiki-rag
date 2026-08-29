"""Hybrid retrieval (Phase 5 placeholder).

Reciprocal-rank fusion combines dense and lexical candidate lists
into a single ranking. Phase 5 implements the real thing; this
module exists now so the package shape matches the plan.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    *ranked_lists: Sequence[tuple[object, float]],
    k: int = 60,
) -> list[tuple[object, float]]:
    """Fuse multiple ranked lists using reciprocal-rank fusion.

    Each input is a list of ``(id, score)`` pairs in descending
    relevance. The same id may appear in multiple lists. RRF adds
    ``1 / (k + rank)`` for each appearance, and returns the union
    ranked by the sum. Final scores are not probabilities; they are
    rank-based and only useful for relative ordering.

    The ``id`` is typed as ``object`` so callers can use strings,
    integers, or other hashables interchangeably. Internally we
    treat the id opaquely (it only needs to be hashable and
    equality-comparable).
    """
    scores: dict[object, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


__all__ = ["reciprocal_rank_fusion"]
