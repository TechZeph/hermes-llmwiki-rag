"""Deterministic query decomposition and multi-query fusion (V3).

No LLM is involved. A question is split into sub-queries only on explicit
coordination cues ("and", "vs", "compared to", "as well as", ";", "?" between
clauses). Each sub-query is retrieved independently with the normal
pipeline and the results are fused with reciprocal-rank fusion, so a
two-part question ("what is X and how does project Y use it") can surface
both pages instead of the one that best matches the whole sentence.

Decomposition is conservative: short questions, single-clause questions
and questions whose parts would be too short are left whole.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Final

from .hybrid import reciprocal_rank_fusion
from .lexical import build_match_query
from .models import Candidate, RetrievalResult

_SPLIT_RE: Final = re.compile(
    r"\s*(?:;|\?\s+(?=\w)|\band also\b|\bas well as\b|\bversus\b|\bvs\.?\b|\bcompared (?:to|with)\b|"
    r"\band (?=(?:how|what|why|when|where|which|who|does|do|is|are|did|can|should)\b))\s*",
    re.IGNORECASE,
)
_MIN_TERMS: Final = 2
_MAX_PARTS: Final = 3


def decompose_query(query: str) -> list[str]:
    """Return sub-queries (2-3) or ``[query]`` when the question is single-clause."""
    text = query.strip()
    if not text or len(text) < 24:
        return [text] if text else []
    parts = [p.strip(" ,.?!") for p in _SPLIT_RE.split(text) if p and p.strip(" ,.?!")]
    parts = [p for p in parts if build_match_query(p).count('"') // 2 >= _MIN_TERMS]
    if len(parts) < 2:
        return [text]
    return parts[:_MAX_PARTS]


def fuse_results(
    query: str,
    parts: Sequence[str],
    results: Sequence[RetrievalResult],
    *,
    top_k: int,
    rrf_k: int = 20,
) -> RetrievalResult:
    """RRF over per-sub-query result lists; candidates keep their first-seen metrics."""
    channels: dict[str, list[int]] = {}
    by_id: dict[int, Candidate] = {}
    for i, res in enumerate(results):
        ordered: list[int] = []
        for cand in res.candidates:
            ordered.append(cand.chunk_id)
            by_id.setdefault(cand.chunk_id, cand)
        channels[f"q{i + 1}"] = ordered
    fused = reciprocal_rank_fusion(channels, k=rrf_k)
    candidates = []
    for entry in fused[:top_k]:
        base = by_id[entry.id]
        parts_hit = ",".join(sorted(entry.ranks))
        candidates.append(
            replace(
                base,
                rrf_score=entry.rrf_score,
                selection_reason=f"multiquery[{parts_hit}] {base.selection_reason}",
            )
        )
    conflicts = tuple(dict.fromkeys(label for res in results for label in res.conflicts))
    return RetrievalResult(
        query=query,
        profile=results[0].profile if results else "answer",
        mode="multiquery",
        candidates=tuple(candidates),
        dense_returned=sum(r.dense_returned for r in results),
        lexical_returned=sum(r.lexical_returned for r in results),
        fused_total=len(fused),
        graph_returned=sum(r.graph_returned for r in results),
        intent=results[0].intent if results else "general",
        conflicts=conflicts,
        elapsed_ms=sum(r.elapsed_ms for r in results),
    )


def retrieve_multiquery(
    query: str,
    retrieve: Callable[[str], RetrievalResult],
    *,
    top_k: int,
    rrf_k: int = 20,
) -> RetrievalResult:
    """Decompose, retrieve each part with ``retrieve``, fuse. Falls back to one call."""
    parts = decompose_query(query)
    if len(parts) <= 1:
        return retrieve(query)
    results = [retrieve(part) for part in parts]
    # The whole question is a fourth voice so a page matching everything still wins.
    results.append(retrieve(query))
    return fuse_results(query, [*parts, query], results, top_k=top_k, rrf_k=rrf_k)


__all__ = ["decompose_query", "fuse_results", "retrieve_multiquery"]
