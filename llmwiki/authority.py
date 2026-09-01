"""Query-intent detection and authority-aware ordering.

Authority is applied as a *stable re-ordering* of already-retrieved
candidates, never as a score mixed into channel metrics. The policy is
deterministic and inspectable:

- Each candidate gets an ``authority_class`` derived from its
  path-derived corpus metadata.
- The query is classified into an intent using explicit keyword rules.
- Candidates whose class is preferred for the intent are promoted within
  a bounded window of the fused ranking; route maps and raw idea drops
  are always demoted below curated knowledge.
- Simple provenance conflicts (several projects competing, or a page
  older than the project's current-state page) are labelled and
  returned, not resolved.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Final

from .models import Candidate

INTENTS: Final = ("current-state", "decision", "chronology", "evidence", "general")

_INTENT_RULES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "chronology",
        re.compile(
            r"\b(when|what happened|timeline|history of|chronolog|on \d{4}-\d{2}-\d{2}|"
            r"last (week|month|session|time)|first (added|introduced|landed)|"
            r"in what order|sequence of events|changelog|log entry)\b"
        ),
    ),
    (
        "decision",
        re.compile(
            r"\b(why|decided|decision|rationale|reason(ing)?|chose|chosen|choice|"
            r"trade-?offs?|justif|instead of|over (using|the alternative)|"
            r"was (it|this) (picked|selected))\b"
        ),
    ),
    (
        "current-state",
        re.compile(
            r"\b(current(ly)?|status|state of|where (are|is) .* (at|now|up to)|"
            r"right now|at the moment|latest|so far|progress|what('s| is) (done|left|"
            r"implemented|working|blocked|next)|blockers?|remaining|outstanding|"
            r"how far along|is .* (finished|complete|done|ready))\b"
        ),
    ),
    (
        "evidence",
        re.compile(
            r"\b(paper|article|transcript|source|according to|what does .* say|"
            r"quote|cite|citation|evidence|original (text|document)|verbatim|"
            r"clipping|raw)\b"
        ),
    ),
)


def detect_intent(query: str) -> str:
    """Classify a query into one of :data:`INTENTS` with explicit rules.

    Rules are checked in a fixed priority order (chronology, decision,
    current-state, evidence) so overlapping cues resolve the same way
    every time. Anything else is ``general``.
    """
    q = query.lower().strip()
    for intent, pattern in _INTENT_RULES:
        if pattern.search(q):
            return intent
    return "general"


def authority_class(candidate: Candidate) -> str:
    """Map path-derived corpus metadata to an authority class label."""
    if candidate.source_kind in ("raw", "clipping"):
        return "idea" if candidate.page_role == "idea" else "evidence"
    if candidate.is_route_map or candidate.page_role == "route-map":
        return "route-map"
    if candidate.page_role in ("current-state", "decision", "log"):
        return candidate.page_role
    if candidate.page_role == "project":
        return "project"
    return "durable"


# Preferred classes per intent. Tier 0 is promoted; tier 1 is neutral;
# tier 2 (route maps, idea drops) is always demoted below the rest.
_TIERS: Final[dict[str, dict[str, int]]] = {
    "current-state": {
        "current-state": 0,
        "durable": 1,
        "project": 1,
        "decision": 1,
        "log": 1,
        "evidence": 1,
    },
    "decision": {
        "decision": 0,
        "durable": 1,
        "project": 1,
        "current-state": 1,
        "log": 1,
        "evidence": 1,
    },
    "chronology": {
        "log": 0,
        "decision": 1,
        "current-state": 1,
        "project": 1,
        "durable": 1,
        "evidence": 1,
    },
    "evidence": {
        "evidence": 0,
        "durable": 1,
        "project": 1,
        "current-state": 1,
        "decision": 1,
        "log": 1,
        "idea": 1,
    },
    "general": {
        "durable": 0,
        "project": 0,
        "current-state": 0,
        "decision": 0,
        "log": 1,
        "evidence": 1,
    },
}
_DEMOTED_TIER: Final = 2
PROMOTION_WINDOW: Final = 20


def _tier(intent: str, klass: str) -> int:
    return _TIERS.get(intent, _TIERS["general"]).get(klass, _DEMOTED_TIER)


def apply_authority_policy(
    candidates: Sequence[Candidate],
    *,
    intent: str,
    profile: str,
    window: int = PROMOTION_WINDOW,
) -> tuple[list[Candidate], tuple[str, ...]]:
    """Label, stably re-order, and flag conflicts among ``candidates``.

    Only candidates inside the first ``window`` positions are eligible for
    promotion so a weakly matched authoritative page cannot leap over
    strongly matched knowledge from deep in the list. Demotion of route
    maps and idea drops applies everywhere. Explicit ``evidence`` and
    ``history`` profiles already scope the corpus, so promotion is a
    no-op there and only labelling/conflict detection runs.
    """
    labelled: list[Candidate] = []
    for rank, cand in enumerate(candidates, start=1):
        klass = authority_class(cand)
        tier = _tier(intent, klass)
        labelled.append(
            replace(
                cand,
                authority_class=klass,
                authority_match=(tier == 0),
                selection_reason=f"{cand.selection_reason} rank={rank}",
            )
        )

    keyed: list[tuple[int, int, Candidate]] = []
    for rank, cand in enumerate(labelled, start=1):
        tier = _tier(intent, cand.authority_class)
        if profile in ("evidence", "history"):
            sort_tier = 1 if tier != _DEMOTED_TIER else _DEMOTED_TIER
        elif tier == 0 and rank > window:
            sort_tier = 1
        else:
            sort_tier = tier
        keyed.append((sort_tier, rank, cand))
    keyed.sort(key=lambda t: (t[0], t[1]))
    ordered = [c for _, _, c in keyed]
    return ordered, detect_conflicts(ordered)


def detect_conflicts(candidates: Sequence[Candidate], *, top_n: int = 10) -> tuple[str, ...]:
    """Return provenance labels for likely contradictions in the top results.

    Two deterministic heuristics:

    - ``competing-projects``: curated project pages from more than one
      project appear in the head of the list.
    - ``older-than-current-state``: a project page in the head is older
      than that project's ``current-state`` page which also appears, so
      its statements may be superseded.
    """
    head = list(candidates[:top_n])
    labels: list[str] = []
    projects = sorted({c.project_id for c in head if c.project_id})
    if len(projects) > 1:
        labels.append("competing-projects: " + ", ".join(projects))
    current_state = {
        c.project_id: c for c in head if c.page_role == "current-state" and c.project_id
    }
    for cand in head:
        cs = current_state.get(cand.project_id or "")
        if cs is None or cand.page_role == "current-state":
            continue
        if cand.updated_at_ns and cs.updated_at_ns and cand.updated_at_ns < cs.updated_at_ns:
            labels.append(f"older-than-current-state: {cand.path} predates {cs.path}")
    # Deduplicate while keeping order.
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            out.append(label)
    return tuple(out)


__all__ = [
    "INTENTS",
    "PROMOTION_WINDOW",
    "apply_authority_policy",
    "authority_class",
    "detect_conflicts",
    "detect_intent",
]
