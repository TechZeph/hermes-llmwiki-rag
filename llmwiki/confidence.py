"""Calibrated retrieve/abstain decisions for automatic injection.

Confidence is never a raw channel score. It is a logistic model over
structural features of a retrieval result (cross-channel agreement, top
margin, authority match, duplicate concentration, ...) fitted on the
labelled dev split and measured on held-out (``docs/evaluation.md``,
Gate A). Without a fitted gate the decision is always "do not inject".
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import RetrievalResult

FEATURE_NAMES: tuple[str, ...] = (
    "rrf_top",
    "rrf_margin",
    "channel_agreement_top5",
    "top1_in_both_channels",
    "authority_top1",
    "dense_top_similarity",
    "bm25_top_norm",
    "n_candidates_norm",
)


def result_features(result: RetrievalResult) -> dict[str, float]:
    """Structural features; every value is finite so the model is total."""
    cands = result.candidates
    top = cands[0] if cands else None
    second = cands[1] if len(cands) > 1 else None
    dense_top = min((c.dense_distance for c in cands if c.dense_distance is not None), default=None)
    bm25_top = max((c.bm25_score for c in cands if c.bm25_score is not None), default=None)
    rrf_top = top.rrf_score if top is not None and top.rrf_score is not None else 0.0
    rrf_second = second.rrf_score if second is not None and second.rrf_score is not None else 0.0
    both = sum(1 for c in cands[:5] if c.dense_rank is not None and c.lexical_rank is not None)
    return {
        "rrf_top": rrf_top,
        "rrf_margin": max(rrf_top - rrf_second, 0.0),
        "channel_agreement_top5": both / 5.0,
        "top1_in_both_channels": (
            1.0
            if top is not None and top.dense_rank is not None and top.lexical_rank is not None
            else 0.0
        ),
        "authority_top1": 1.0 if top is not None and top.authority_match else 0.0,
        # sqlite-vec cosine distance in [0, 2]; map to a similarity in [0, 1].
        "dense_top_similarity": (1.0 - dense_top / 2.0) if dense_top is not None else 0.0,
        # BM25 scores are unbounded; squash so one huge value cannot dominate.
        "bm25_top_norm": (1.0 - math.exp(-bm25_top / 10.0))
        if bm25_top is not None and bm25_top > 0
        else 0.0,
        "n_candidates_norm": min(len(cands), 10) / 10.0,
    }


@dataclass(frozen=True, slots=True)
class InjectionGate:
    weights: dict[str, float]
    bias: float
    threshold: float
    fitted_on: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def score(self, features: Mapping[str, float]) -> float:
        z = self.bias + sum(
            self.weights.get(k, 0.0) * float(features.get(k, 0.0)) for k in FEATURE_NAMES
        )
        z = max(min(z, 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(-z))

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "bias": self.bias,
            "threshold": self.threshold,
            "fitted_on": self.fitted_on,
            "metrics": dict(self.metrics),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> InjectionGate:
        return InjectionGate(
            weights={str(k): float(v) for k, v in dict(data.get("weights", {})).items()},
            bias=float(data.get("bias", 0.0)),
            threshold=float(data.get("threshold", 0.5)),
            fitted_on=str(data.get("fitted_on", "")),
            metrics=dict(data.get("metrics", {})),
        )


@dataclass(frozen=True, slots=True)
class Decision:
    inject: bool
    score: float
    threshold: float
    reason: str
    features: dict[str, float]


def load_gate(path: Path) -> InjectionGate | None:
    if not path.is_file():
        return None
    try:
        return InjectionGate.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def decide(result: RetrievalResult, gate: InjectionGate | None) -> Decision:
    features = result_features(result)
    if gate is None:
        return Decision(False, 0.0, 1.0, "no-calibrated-gate", features)
    if not result.candidates:
        return Decision(False, 0.0, gate.threshold, "no-candidates", features)
    score = gate.score(features)
    if score >= gate.threshold:
        return Decision(True, score, gate.threshold, "above-threshold", features)
    return Decision(False, score, gate.threshold, "below-threshold", features)


# --- fitting (used by the evaluation harness, numpy only) -------------------


def fit_gate(
    samples: Sequence[tuple[Mapping[str, float], bool]],
    *,
    l2: float = 0.05,
    epochs: int = 3000,
    learning_rate: float = 0.5,
    fitted_on: str = "",
) -> InjectionGate:
    """Fit an L2-regularised logistic regression by full-batch gradient descent.

    ``samples`` are ``(features, label)`` pairs where the label is True when
    injecting would have been correct (top result relevant). Features are
    standardised internally; the returned weights are expressed on the raw
    feature scale so scoring needs no extra state.
    """
    import numpy as np

    if not samples:
        raise ValueError("cannot fit a gate on zero samples")
    x = np.array([[float(f.get(k, 0.0)) for k in FEATURE_NAMES] for f, _ in samples], dtype=float)
    y = np.array([1.0 if label else 0.0 for _, label in samples], dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    xs = (x - mean) / std
    w = np.zeros(xs.shape[1])
    b = 0.0
    n = len(y)
    # Balance classes so a rare abstain class still shapes the boundary.
    pos = max(y.sum(), 1.0)
    neg = max(n - y.sum(), 1.0)
    sample_w = np.where(y == 1.0, n / (2 * pos), n / (2 * neg))
    for _ in range(epochs):
        z = xs @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        grad = (p - y) * sample_w
        gw = xs.T @ grad / n + l2 * w
        gb = grad.mean()
        w -= learning_rate * gw
        b -= learning_rate * gb
    raw_w = w / std
    raw_b = float(b - (raw_w * mean).sum())
    return InjectionGate(
        weights={k: float(v) for k, v in zip(FEATURE_NAMES, raw_w, strict=True)},
        bias=raw_b,
        threshold=0.5,
        fitted_on=fitted_on,
    )


def choose_threshold(
    gate: InjectionGate,
    samples: Sequence[tuple[Mapping[str, float], bool]],
    *,
    min_precision: float,
) -> float:
    """Lowest threshold whose precision on ``samples`` meets ``min_precision``.

    Falls back to the threshold with the best precision when none reaches
    the target, so the caller can see (and reject) the shortfall.
    """
    scored = sorted(((gate.score(f), label) for f, label in samples), key=lambda t: -t[0])
    best_t, best_p = 1.01, -1.0
    for i in range(len(scored)):
        threshold = scored[i][0]
        selected = [label for s, label in scored if s >= threshold]
        precision = sum(1 for label in selected if label) / len(selected)
        if precision >= min_precision:
            best_t, best_p = threshold, precision
        elif precision > best_p and best_t > 1.0:
            best_p = precision
    return best_t if best_t <= 1.0 else max(s for s, _ in scored)


def gate_report(
    gate: InjectionGate,
    samples: Sequence[tuple[Mapping[str, float], bool, bool]],
) -> dict[str, float]:
    """Precision / coverage / abstain-rate / pollution for ``(features, relevant, is_abstain_question)``."""
    injected = [(rel, ab) for f, rel, ab in samples if gate.score(f) >= gate.threshold]
    retrieve_qs = [1 for _, _, ab in samples if not ab]
    abstain_qs = [1 for _, _, ab in samples if ab]
    injected_on_abstain = sum(1 for _, ab in injected if ab)
    injected_on_retrieve = [rel for rel, ab in injected if not ab]
    precision = (
        (sum(1 for rel in injected_on_retrieve if rel) / len(injected_on_retrieve))
        if injected_on_retrieve
        else 0.0
    )
    coverage = (len(injected_on_retrieve) / len(retrieve_qs)) if retrieve_qs else 0.0
    abstain_rate = (1.0 - injected_on_abstain / len(abstain_qs)) if abstain_qs else 1.0
    pollution = (
        ((sum(1 for rel in injected_on_retrieve if not rel) + injected_on_abstain) / len(injected))
        if injected
        else 0.0
    )
    return {
        "precision": precision,
        "coverage": coverage,
        "abstain_rate": abstain_rate,
        "pollution": pollution,
        "injected": float(len(injected)),
        "n": float(len(samples)),
    }


__all__ = [
    "FEATURE_NAMES",
    "Decision",
    "InjectionGate",
    "choose_threshold",
    "decide",
    "fit_gate",
    "gate_report",
    "load_gate",
    "result_features",
]
