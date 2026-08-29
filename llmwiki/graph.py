"""Obsidian wikilink graph (Phase 10 placeholder)."""

from __future__ import annotations

from collections.abc import Iterable


def expand_via_wikilinks(
    seed_ids: Iterable[int],
    *,
    max_hops: int = 1,
) -> set[int]:
    """Placeholder. Returns just the seed set.

    Phase 10 will traverse the graph stored in ``graph_edges`` and
    return the union of nodes reachable within ``max_hops``.
    """
    return set(seed_ids)


__all__ = ["expand_via_wikilinks"]
