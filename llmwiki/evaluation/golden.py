"""Golden question sets: schema, loading, validation, and drafting merges."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

CATEGORIES: Final = (
    "current-state",
    "decision",
    "exact-term",
    "concept",
    "evidence",
    "chronology",
    "ambiguity",
    "no-answer",
)
SPLITS: Final = ("dev", "heldout")
AUTHORITY_CLASSES: Final = (
    "current-state",
    "decision",
    "durable",
    "evidence",
    "log",
    "idea",
    "none",
)
MODES: Final = ("retrieve", "abstain")
_HEADING_RE: Final = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class RelevantSource:
    path: str
    sections: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    category: str
    split: str
    query: str
    profile: str
    authority_class: str
    mode: str
    relevant: tuple[RelevantSource, ...] = ()
    notes: str = ""

    @property
    def relevant_paths(self) -> frozenset[str]:
        return frozenset(r.path for r in self.relevant)


@dataclass(frozen=True, slots=True)
class GoldenSet:
    corpus: str
    version: str
    questions: tuple[Question, ...]
    source_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def subset(self, *, split: str | None = None, category: str | None = None) -> GoldenSet:
        qs = tuple(
            q
            for q in self.questions
            if (split is None or q.split == split) and (category is None or q.category == category)
        )
        return GoldenSet(self.corpus, self.version, qs, self.source_path, dict(self.extra))

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for q in self.questions:
            bucket = out.setdefault(q.category, {"dev": 0, "heldout": 0})
            bucket[q.split] = bucket.get(q.split, 0) + 1
        return out


def _question_from_dict(raw: dict[str, Any]) -> Question:
    relevant = tuple(
        RelevantSource(
            path=str(r["path"]), sections=tuple(str(s) for s in r.get("sections", []) or [])
        )
        for r in raw.get("relevant", []) or []
    )
    return Question(
        id=str(raw["id"]),
        category=str(raw["category"]),
        split=str(raw["split"]),
        query=str(raw["query"]),
        profile=str(raw["profile"]),
        authority_class=str(raw["authority_class"]),
        mode=str(raw["mode"]),
        relevant=relevant,
        notes=str(raw.get("notes", "") or ""),
    )


def load_golden(path: Path) -> GoldenSet:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = tuple(_question_from_dict(q) for q in data.get("questions", []))
    extra = {k: v for k, v in data.items() if k not in {"corpus", "version", "questions"}}
    return GoldenSet(
        corpus=str(data.get("corpus", "")),
        version=str(data.get("version", "")),
        questions=questions,
        source_path=path,
        extra=extra,
    )


def _headings_of(file: Path) -> set[str]:
    try:
        text = file.read_text(encoding="utf-8")
    except OSError:
        return set()
    return {m.group(2).strip() for m in _HEADING_RE.finditer(text)}


def validate_golden(golden: GoldenSet, *, vault: Path | None = None) -> list[str]:
    """Return human-readable problems; an empty list means the set is valid."""
    problems: list[str] = []
    seen_ids: set[str] = set()
    for q in golden.questions:
        prefix = f"{q.id}: "
        if q.id in seen_ids:
            problems.append(prefix + "duplicate id")
        seen_ids.add(q.id)
        if q.category not in CATEGORIES:
            problems.append(prefix + f"unknown category {q.category!r}")
        if q.split not in SPLITS:
            problems.append(prefix + f"unknown split {q.split!r}")
        if q.authority_class not in AUTHORITY_CLASSES:
            problems.append(prefix + f"unknown authority_class {q.authority_class!r}")
        if q.mode not in MODES:
            problems.append(prefix + f"unknown mode {q.mode!r}")
        if not q.query.strip():
            problems.append(prefix + "empty query")
        valid_profile = q.profile in ("answer", "evidence", "history", "all", "current") or (
            q.profile.startswith("project:") and len(q.profile) > len("project:")
        )
        if not valid_profile:
            problems.append(prefix + f"invalid profile {q.profile!r}")
        if q.mode == "abstain":
            if q.relevant:
                problems.append(prefix + "abstain questions must have no relevant sources")
            if q.authority_class != "none":
                problems.append(prefix + "abstain questions must use authority_class 'none'")
        elif not q.relevant:
            problems.append(prefix + "retrieve questions need at least one relevant source")
        for rel in q.relevant:
            if rel.path.startswith("/") or "\\" in rel.path or ".." in rel.path.split("/"):
                problems.append(prefix + f"path must be vault-relative: {rel.path}")
            if vault is not None:
                file = vault / rel.path
                if not file.is_file():
                    problems.append(prefix + f"missing file {rel.path}")
                    continue
                if rel.sections:
                    headings = _headings_of(file)
                    for section in rel.sections:
                        if section not in headings:
                            problems.append(prefix + f"heading {section!r} not found in {rel.path}")
    return problems


def merge_drafts(paths: Iterable[Path], *, corpus: str, version: str) -> dict[str, Any]:
    """Combine draft files into one golden-set document (unvalidated)."""
    questions: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        questions.extend(data.get("questions", []))
    questions.sort(
        key=lambda q: (
            CATEGORIES.index(q["category"]) if q["category"] in CATEGORIES else 99,
            q["id"],
        )
    )
    return {"corpus": corpus, "version": version, "questions": questions}


def stratification_report(golden: GoldenSet, *, minimum_total: int = 60) -> list[str]:
    """Check the Stage 0 stratification requirement; returns problems."""
    problems: list[str] = []
    counts = golden.counts()
    total = len(golden.questions)
    if total < minimum_total:
        problems.append(f"only {total} questions; need at least {minimum_total}")
    for category in CATEGORIES:
        c = counts.get(category, {"dev": 0, "heldout": 0})
        if c["dev"] + c["heldout"] == 0:
            problems.append(f"category {category!r} has no questions")
        elif c["heldout"] == 0:
            problems.append(f"category {category!r} has no held-out questions")
    return problems


def write_golden(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def question_ids(questions: Sequence[Question]) -> list[str]:
    return [q.id for q in questions]


__all__ = [
    "AUTHORITY_CLASSES",
    "CATEGORIES",
    "MODES",
    "SPLITS",
    "GoldenSet",
    "Question",
    "RelevantSource",
    "load_golden",
    "merge_drafts",
    "question_ids",
    "stratification_report",
    "validate_golden",
    "write_golden",
]
