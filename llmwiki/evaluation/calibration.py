"""Stage 4 calibration: routing accuracy and the automatic-injection gate.

Fits the logistic injection gate on the dev split, chooses the lowest
threshold meeting the predeclared precision target on dev, then reports
precision / coverage / abstain rate / context pollution on held-out
(``docs/evaluation.md``, Gate A). Also measures the deterministic router
against the golden profiles.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..confidence import (
    InjectionGate,
    choose_threshold,
    fit_gate,
    gate_report,
    result_features,
)
from ..models import RetrievalResult
from ..routing import route_query
from .golden import GoldenSet, Question
from .metrics import candidate_matches

GATE_A = {
    "min_precision": 0.90,
    "min_abstain_rate": 0.80,
    "min_coverage": 0.60,
    "max_pollution": 0.10,
}

RetrieveFn = Callable[[Question], RetrievalResult]


def _top_relevant(question: Question, result: RetrievalResult) -> bool:
    if question.mode == "abstain" or not result.candidates:
        return False
    top = result.candidates[0]
    return any(candidate_matches(top, rel) for rel in question.relevant)


def _context_relevant(question: Question, result: RetrievalResult, *, head: int = 4) -> bool:
    if question.mode == "abstain":
        return False
    return any(
        candidate_matches(c, rel) for c in result.candidates[:head] for rel in question.relevant
    )


@dataclass(slots=True)
class RoutingReport:
    n: int
    retrieve_routed_recall: float
    profile_accuracy: float
    abstain_routed_to_retrieve: float
    by_reason: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "retrieve_routed_recall": self.retrieve_routed_recall,
            "profile_accuracy": self.profile_accuracy,
            "abstain_routed_to_retrieve": self.abstain_routed_to_retrieve,
            "by_reason": dict(self.by_reason),
        }


def evaluate_routing(golden: GoldenSet, *, known_projects: Sequence[str] = ()) -> RoutingReport:
    routed_ok = 0
    profile_ok = 0
    retrieve_n = 0
    abstain_n = 0
    abstain_routed = 0
    reasons: dict[str, int] = {}
    for q in golden.questions:
        route = route_query(q.query, known_projects=known_projects)
        reasons[route.reason] = reasons.get(route.reason, 0) + 1
        if q.mode == "retrieve":
            retrieve_n += 1
            if route.retrieve:
                routed_ok += 1
                if route.profile == q.profile:
                    profile_ok += 1
        else:
            abstain_n += 1
            if route.retrieve:
                abstain_routed += 1
    return RoutingReport(
        n=len(golden.questions),
        retrieve_routed_recall=(routed_ok / retrieve_n) if retrieve_n else 0.0,
        profile_accuracy=(profile_ok / routed_ok) if routed_ok else 0.0,
        abstain_routed_to_retrieve=(abstain_routed / abstain_n) if abstain_n else 0.0,
        by_reason=reasons,
    )


def collect_samples(
    golden: GoldenSet, retrieve: RetrieveFn
) -> list[tuple[dict[str, float], bool, bool, bool]]:
    """``(features, top_relevant, is_abstain, context_relevant)`` per question."""
    samples: list[tuple[dict[str, float], bool, bool, bool]] = []
    for q in golden.questions:
        result = retrieve(q)
        samples.append(
            (
                result_features(result),
                _top_relevant(q, result),
                q.mode == "abstain",
                _context_relevant(q, result),
            )
        )
    return samples


def calibrate(
    golden: GoldenSet,
    retrieve: RetrieveFn,
    *,
    known_projects: Sequence[str] = (),
    fitted_on: str = "",
) -> tuple[InjectionGate, dict[str, Any]]:
    dev = golden.subset(split="dev")
    held = golden.subset(split="heldout")
    dev_samples = collect_samples(dev, retrieve)
    held_samples = collect_samples(held, retrieve)

    fit_pairs = [(f, top) for f, top, _, _ in dev_samples]
    gate = fit_gate(fit_pairs, fitted_on=fitted_on)
    threshold = choose_threshold(gate, fit_pairs, min_precision=GATE_A["min_precision"])
    gate = InjectionGate(gate.weights, gate.bias, threshold, gate.fitted_on)

    dev_report = gate_report(gate, [(f, top, ab) for f, top, ab, _ in dev_samples])
    held_report = gate_report(gate, [(f, top, ab) for f, top, ab, _ in held_samples])
    # Context pollution uses the injected head (top 4), not only the top-1.
    held_injected = [(ab, ctx) for f, _, ab, ctx in held_samples if gate.score(f) >= threshold]
    context_pollution = (
        sum(1 for ab, ctx in held_injected if ab or not ctx) / len(held_injected)
        if held_injected
        else 0.0
    )
    held_report["context_pollution"] = context_pollution

    # Safety clauses decide whether opt-in injection is permitted at all;
    # the coverage clause additionally decides whether default-on could
    # ever be recommended (it never is in V1.1).
    safety_passed = (
        held_report["precision"] >= GATE_A["min_precision"]
        and held_report["abstain_rate"] >= GATE_A["min_abstain_rate"]
        and context_pollution <= GATE_A["max_pollution"]
    )
    gate_a_passed = safety_passed and held_report["coverage"] >= GATE_A["min_coverage"]
    routing = {
        "dev": evaluate_routing(dev, known_projects=known_projects).to_dict(),
        "heldout": evaluate_routing(held, known_projects=known_projects).to_dict(),
    }
    metrics = {
        "gate_a": dict(GATE_A),
        "gate_a_passed": gate_a_passed,
        "safety_passed": safety_passed,
        "threshold": threshold,
        "dev": dev_report,
        "heldout": held_report,
        "routing": routing,
        "calibrated_at": datetime.now(UTC).isoformat(),
        "fitted_on": fitted_on,
    }
    gate = InjectionGate(gate.weights, gate.bias, threshold, fitted_on, metrics)
    return gate, metrics


def write_gate(gate: InjectionGate, path: Path) -> None:
    path.write_text(json.dumps(gate.to_dict(), indent=2) + "\n", encoding="utf-8")


__all__ = [
    "GATE_A",
    "RoutingReport",
    "calibrate",
    "collect_samples",
    "evaluate_routing",
    "write_gate",
]
