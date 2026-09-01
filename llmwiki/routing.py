"""Deterministic query routing: should we retrieve, and from which profile?

Explicit retrieval modes (tool calls) always override this router; it
exists for the opt-in automatic path where the only input is the current
user message. Every decision carries a human-readable reason so routing
can be evaluated and audited without an LLM.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from .authority import detect_intent
from .lexical import build_match_query

_QUESTION_WORDS: Final = frozenset(
    [
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "does",
        "do",
        "did",
        "is",
        "are",
        "was",
        "were",
        "can",
        "could",
        "should",
        "would",
        "will",
    ]
)
_GREETING_RE: Final = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|cheers|good (morning|evening|night)|yes|no|sure)\b[\s!.]*$",
    re.IGNORECASE,
)
_EXPLICIT_WIKI_RE: Final = re.compile(
    r"\b(in|from|check|search|look up|consult|per|according to) (the |my |our )?(wiki|vault|knowledge ?base|notes)\b",
    re.IGNORECASE,
)
_HISTORY_RE: Final = re.compile(
    r"\b(in the log|history|chronolog|what happened on|timeline)\b", re.IGNORECASE
)
_EVIDENCE_RE: Final = re.compile(
    r"\b(paper|article|transcript|clipping|raw source|original source|what does the .* say)\b",
    re.IGNORECASE,
)
_PROJECT_RE: Final = re.compile(r"\bproject[: ]+([A-Za-z0-9][A-Za-z0-9._-]*)", re.IGNORECASE)
_CODE_BLOCK_RE: Final = re.compile(r"```")
_COMMAND_RE: Final = re.compile(
    r"^\s*(run|execute|open|write|create|delete|install|git |ls |cd |cat |edit|fix|refactor|deploy|commit|push)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Route:
    retrieve: bool
    profile: str
    intent: str
    reason: str
    explicit: bool = False


def informative_terms(query: str) -> int:
    """Number of non-stopword terms the lexical channel would use."""
    match = build_match_query(query)
    return 0 if not match else match.count('"') // 2


def route_query(
    query: str,
    *,
    default_profile: str = "answer",
    known_projects: Sequence[str] = (),
) -> Route:
    text = query.strip()
    intent = detect_intent(text)
    if not text:
        return Route(False, default_profile, intent, "empty")
    if _GREETING_RE.match(text):
        return Route(False, default_profile, intent, "greeting-or-ack")
    if _CODE_BLOCK_RE.search(text):
        return Route(False, default_profile, intent, "contains-code-block")
    if len(text) > 2000:
        return Route(False, default_profile, intent, "too-long")
    terms = informative_terms(text)
    if terms < 2:
        return Route(False, default_profile, intent, "too-few-terms")

    lowered = text.lower()
    project_hit = None
    m = _PROJECT_RE.search(text)
    if m and (not known_projects or m.group(1) in known_projects):
        project_hit = m.group(1)
    else:
        for pid in known_projects:
            if pid and pid.lower() in lowered:
                project_hit = pid
                break

    explicit = bool(_EXPLICIT_WIKI_RE.search(text))
    if _HISTORY_RE.search(text) or intent == "chronology":
        return Route(True, "history", intent, "chronology-cue", explicit)
    if _EVIDENCE_RE.search(text) or intent == "evidence":
        return Route(True, "evidence", intent, "evidence-cue", explicit)
    if project_hit:
        return Route(True, f"project:{project_hit}", intent, f"project-cue:{project_hit}", explicit)
    if explicit:
        return Route(True, default_profile, intent, "explicit-wiki-reference", True)
    if _COMMAND_RE.match(text):
        return Route(False, default_profile, intent, "imperative-command")
    first = lowered.split()[0].strip("?,.!") if lowered.split() else ""
    if text.endswith("?") or first in _QUESTION_WORDS or intent != "general":
        return Route(True, default_profile, intent, "question-like")
    if terms >= 6:
        return Route(True, default_profile, intent, "long-informative-statement")
    return Route(False, default_profile, intent, "no-retrieval-cue")


__all__ = ["Route", "informative_terms", "route_query"]
