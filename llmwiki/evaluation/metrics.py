"""Retrieval metrics computed per question and aggregated per run.

All metrics are document-level unless a question pins ``sections``, in
which case a chunk counts only when its heading breadcrumb or section
name matches one of the pinned headings. Documents are deduplicated in
rank order before computing rank-based metrics so a page that returns
three chunks is credited once, at its best rank.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..models import Candidate, RetrievalResult
from .golden import Question, RelevantSource

RECALL_KS: tuple[int, ...] = (1, 3, 5, 10)

# Candidate authority classes that satisfy each golden authority class.
_AUTHORITY_SATISFIES: dict[str, frozenset[str]] = {
    "current-state": frozenset({"current-state"}),
    "decision": frozenset({"decision"}),
    "durable": frozenset({"durable", "project"}),
    "evidence": frozenset({"evidence"}),
    "idea": frozenset({"idea"}),
    "log": frozenset({"log"}),
}


def candidate_matches(candidate: Candidate, relevant: RelevantSource) -> bool:
    if candidate.path != relevant.path:
        return False
    if not relevant.sections:
        return True
    wanted = set(relevant.sections)
    return candidate.section_name in wanted or any(h in wanted for h in candidate.heading_path)


def _dedupe_documents(candidates: Sequence[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    out: list[Candidate] = []
    for c in candidates:
        if c.path in seen:
            continue
        seen.add(c.path)
        out.append(c)
    return out


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


@dataclass(slots=True)
class QuestionOutcome:
    id: str
    category: str
    split: str
    mode: str
    profile: str
    query: str
    intent: str
    retrieved: list[dict[str, Any]]
    relevant_paths: list[str]
    hit_ranks: list[int]
    first_hit_rank: int | None
    hit_at: dict[str, float]
    recall_at: dict[str, float]
    reciprocal_rank: float
    ndcg_at_10: float
    authority_top1_match: bool | None
    authority_any_top3: bool | None
    duplicate_concentration: float
    citation_ok_fraction: float
    conflicts: list[str]
    elapsed_ms: float
    features: dict[str, float | None]
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _features(result: RetrievalResult) -> dict[str, float | None]:
    """Signals for later confidence calibration; never a confidence by themselves."""
    cands = result.candidates
    top = cands[0] if cands else None
    second = cands[1] if len(cands) > 1 else None
    dense_top = min((c.dense_distance for c in cands if c.dense_distance is not None), default=None)
    bm25_top = max((c.bm25_score for c in cands if c.bm25_score is not None), default=None)
    rrf_top = top.rrf_score if top is not None else None
    rrf_second = second.rrf_score if second is not None else None
    both = sum(1 for c in cands[:5] if c.dense_rank is not None and c.lexical_rank is not None)
    return {
        "dense_top_distance": dense_top,
        "bm25_top_score": bm25_top,
        "rrf_top": rrf_top,
        "rrf_margin": (rrf_top - rrf_second)
        if rrf_top is not None and rrf_second is not None
        else None,
        "channel_agreement_top5": float(both) / 5.0 if cands else None,
        "top1_in_both_channels": (
            1.0
            if top is not None and top.dense_rank is not None and top.lexical_rank is not None
            else 0.0
        )
        if cands
        else None,
        "rerank_top": top.rerank_score if top is not None else None,
        "authority_top1": 1.0
        if top is not None and top.authority_match
        else (0.0 if top else None),
        "n_candidates": float(len(cands)),
        "n_conflicts": float(len(result.conflicts)),
    }


def score_question(
    question: Question,
    result: RetrievalResult,
    *,
    citation_ok: Sequence[bool],
    top_n: int = 10,
) -> QuestionOutcome:
    cands = list(result.candidates)
    docs = _dedupe_documents(cands)
    relevant = list(question.relevant)
    relevant_paths = sorted({r.path for r in relevant})

    # Rank (1-based, document-deduplicated) at which each relevant path first appears.
    hit_ranks: list[int] = []
    matched_paths: dict[str, int] = {}
    for rank, cand in enumerate(docs, start=1):
        for rel in relevant:
            if rel.path in matched_paths:
                continue
            # A section-pinned relevant source can be satisfied by a later
            # chunk of the same document, so check all chunks of that doc.
            same_doc = [c for c in cands if c.path == cand.path]
            if any(candidate_matches(c, rel) for c in same_doc):
                matched_paths[rel.path] = rank
                hit_ranks.append(rank)
    hit_ranks.sort()
    first = hit_ranks[0] if hit_ranks else None

    n_rel = len(relevant_paths)
    hit_at = {str(k): (1.0 if first is not None and first <= k else 0.0) for k in RECALL_KS}
    recall_at = {
        str(k): (sum(1 for r in hit_ranks if r <= k) / n_rel if n_rel else 0.0) for k in RECALL_KS
    }
    rr = (1.0 / first) if first else 0.0
    gains = [1.0 if rank in hit_ranks else 0.0 for rank in range(1, min(len(docs), 10) + 1)]
    ideal = [1.0] * min(n_rel, 10)
    ndcg = (_dcg(gains) / _dcg(ideal)) if ideal and _dcg(ideal) > 0 else 0.0

    if question.mode == "abstain" or not cands:
        auth_top1: bool | None = None
        auth_any3: bool | None = None
    else:
        satisfied = _AUTHORITY_SATISFIES.get(question.authority_class, frozenset())
        auth_top1 = cands[0].authority_class in satisfied
        auth_any3 = any(c.authority_class in satisfied for c in cands[:3])

    head = cands[:top_n]
    if head:
        counts: dict[int, int] = {}
        for c in head:
            counts[c.document_id] = counts.get(c.document_id, 0) + 1
        dup = max(counts.values()) / len(head)
    else:
        dup = 0.0
    cite_ok = (sum(1 for ok in citation_ok if ok) / len(citation_ok)) if citation_ok else 1.0

    return QuestionOutcome(
        id=question.id,
        category=question.category,
        split=question.split,
        mode=question.mode,
        profile=question.profile,
        query=question.query,
        intent=result.intent,
        retrieved=[
            {
                "rank": i + 1,
                "path": c.path,
                "chunk_id": c.chunk_id,
                "section": c.section_name,
                "heading_path": list(c.heading_path),
                "authority_class": c.authority_class,
                "dense_rank": c.dense_rank,
                "dense_distance": c.dense_distance,
                "lexical_rank": c.lexical_rank,
                "bm25_score": c.bm25_score,
                "rrf_score": c.rrf_score,
                "rerank_score": c.rerank_score,
            }
            for i, c in enumerate(head)
        ],
        relevant_paths=relevant_paths,
        hit_ranks=hit_ranks,
        first_hit_rank=first,
        hit_at=hit_at,
        recall_at=recall_at,
        reciprocal_rank=rr,
        ndcg_at_10=ndcg,
        authority_top1_match=auth_top1,
        authority_any_top3=auth_any3,
        duplicate_concentration=dup,
        citation_ok_fraction=cite_ok,
        conflicts=list(result.conflicts),
        elapsed_ms=result.elapsed_ms,
        features=_features(result),
    )


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


@dataclass(slots=True)
class Aggregate:
    n: int
    n_retrieve: int
    n_abstain: int
    hit_at: dict[str, float | None]
    recall_at: dict[str, float | None]
    mrr: float | None
    ndcg_at_10: float | None
    authority_accuracy_top1: float | None
    authority_any_top3: float | None
    duplicate_concentration: float | None
    citation_fidelity: float | None
    conflict_rate: float | None
    abstain_mean_top_rrf: float | None
    abstain_mean_top_dense_distance: float | None
    retrieve_mean_top_rrf: float | None
    retrieve_mean_top_dense_distance: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    errors: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate(outcomes: Sequence[QuestionOutcome]) -> Aggregate:
    retrieve = [o for o in outcomes if o.mode == "retrieve" and not o.error]
    abstain = [o for o in outcomes if o.mode == "abstain" and not o.error]
    auth = [o.authority_top1_match for o in retrieve if o.authority_top1_match is not None]
    auth3 = [o.authority_any_top3 for o in retrieve if o.authority_any_top3 is not None]
    latencies = [o.elapsed_ms for o in outcomes if not o.error]

    def feat(items: Sequence[QuestionOutcome], key: str) -> float | None:
        vals = [v for o in items if (v := o.features.get(key)) is not None]
        return _mean(vals)

    return Aggregate(
        n=len(outcomes),
        n_retrieve=len(retrieve),
        n_abstain=len(abstain),
        hit_at={str(k): _mean([o.hit_at[str(k)] for o in retrieve]) for k in RECALL_KS},
        recall_at={str(k): _mean([o.recall_at[str(k)] for o in retrieve]) for k in RECALL_KS},
        mrr=_mean([o.reciprocal_rank for o in retrieve]),
        ndcg_at_10=_mean([o.ndcg_at_10 for o in retrieve]),
        authority_accuracy_top1=_mean([1.0 if a else 0.0 for a in auth]),
        authority_any_top3=_mean([1.0 if a else 0.0 for a in auth3]),
        duplicate_concentration=_mean([o.duplicate_concentration for o in retrieve]),
        citation_fidelity=_mean([o.citation_ok_fraction for o in outcomes if not o.error]),
        conflict_rate=_mean([1.0 if o.conflicts else 0.0 for o in retrieve]),
        abstain_mean_top_rrf=feat(abstain, "rrf_top"),
        abstain_mean_top_dense_distance=feat(abstain, "dense_top_distance"),
        retrieve_mean_top_rrf=feat(retrieve, "rrf_top"),
        retrieve_mean_top_dense_distance=feat(retrieve, "dense_top_distance"),
        latency_p50_ms=_percentile(latencies, 0.5),
        latency_p95_ms=_percentile(latencies, 0.95),
        errors=sum(1 for o in outcomes if o.error),
    )


def per_category(outcomes: Sequence[QuestionOutcome]) -> dict[str, Aggregate]:
    groups: dict[str, list[QuestionOutcome]] = {}
    for o in outcomes:
        groups.setdefault(o.category, []).append(o)
    return {cat: aggregate(items) for cat, items in sorted(groups.items())}


def corpus_fingerprint(rows: Sequence[tuple[str, str]]) -> str:
    """Stable hash of ``(path, content_hash)`` rows describing the indexed corpus."""
    h = hashlib.sha256()
    for path, content_hash in sorted(rows):
        h.update(path.encode("utf-8"))
        h.update(b"\t")
        h.update(content_hash.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def stdev(values: Sequence[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else None


__all__ = [
    "RECALL_KS",
    "Aggregate",
    "QuestionOutcome",
    "aggregate",
    "candidate_matches",
    "corpus_fingerprint",
    "per_category",
    "score_question",
    "stdev",
]
