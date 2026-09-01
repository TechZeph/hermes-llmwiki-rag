"""Reciprocal-rank fusion of independent retrieval channels.

RRF is candidate fusion, not confidence: the fused score only orders the
union of the channel lists. Raw per-channel metrics and ranks are kept
on the typed candidate so downstream policy, calibration, and evaluation
can reason about them separately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FusedEntry:
    """One fused id with its per-channel ranks and RRF score."""

    id: int
    rrf_score: float
    ranks: Mapping[str, int]


def reciprocal_rank_fusion(
    channels: Mapping[str, Sequence[int]],
    *,
    k: int = 60,
) -> list[FusedEntry]:
    """Fuse ordered id lists from named channels.

    ``channels`` maps a channel name (``"dense"``, ``"lexical"``) to its
    ids in best-first order. Each appearance adds ``1 / (k + rank)`` with
    ranks starting at 1. Ties are broken by the best single-channel rank,
    then by id, so results are deterministic.
    """
    if k < 0:
        raise ValueError("rrf k must be non-negative")
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, ordered in channels.items():
        for rank, doc_id in enumerate(ordered, start=1):
            key = int(doc_id)
            if key in ranks and name in ranks[key]:
                continue  # a channel lists an id at most once
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(key, {})[name] = rank
    fused = [FusedEntry(id=i, rrf_score=s, ranks=dict(ranks[i])) for i, s in scores.items()]
    fused.sort(key=lambda e: (-e.rrf_score, min(e.ranks.values()), e.id))
    return fused


def diversify(
    ordered: Sequence[int],
    group_of: Mapping[int, int],
    *,
    max_per_group: int,
) -> list[int]:
    """Keep order but admit at most ``max_per_group`` ids per group.

    Used for document diversification: a single long page should not
    fill every slot of the final list. ``max_per_group <= 0`` disables
    the cap.
    """
    if max_per_group <= 0:
        return list(ordered)
    seen: dict[int, int] = {}
    kept: list[int] = []
    for item in ordered:
        group = group_of.get(item, item)
        n = seen.get(group, 0)
        if n >= max_per_group:
            continue
        seen[group] = n + 1
        kept.append(item)
    return kept


__all__ = ["FusedEntry", "diversify", "reciprocal_rank_fusion"]
