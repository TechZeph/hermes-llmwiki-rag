"""Deterministic routing and the calibrated injection gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmwiki.confidence import (
    FEATURE_NAMES,
    InjectionGate,
    choose_threshold,
    decide,
    fit_gate,
    gate_report,
    load_gate,
    result_features,
)
from llmwiki.models import Candidate, RetrievalResult
from llmwiki.routing import informative_terms, route_query


@pytest.mark.parametrize(
    ("query", "retrieve", "profile", "reason_prefix"),
    [
        ("", False, "answer", "empty"),
        ("thanks!", False, "answer", "greeting"),
        ("hi", False, "answer", "greeting"),
        ("ok", False, "answer", "greeting"),
        ("sqlite", False, "answer", "too-few-terms"),
        ("run the tests and fix the failures", False, "answer", "imperative"),
        ("here is code ```print(1)``` what does it do?", False, "answer", "contains-code"),
        ("What is the current status of the RAG project?", True, "answer", "question-like"),
        ("why did we pick sqlite-vec over faiss", True, "answer", "question-like"),
        ("when did the sqlite-vec projection land?", True, "history", "chronology"),
        ("what does the RRF paper say about Condorcet?", True, "evidence", "evidence"),
        (
            "summarise the decisions for project:hermes-llmwiki-rag",
            True,
            "project:hermes-llmwiki-rag",
            "project-cue",
        ),
        ("check the wiki for the hosp-core menu ingestion notes", True, "answer", "explicit-wiki"),
        (
            "the hospitality platform pricing model and QR menu flow need review before launch",
            True,
            "answer",
            "long-informative",
        ),
        ("nice weather today", False, "answer", "no-retrieval-cue"),
    ],
)
def test_route_query_rules(query: str, retrieve: bool, profile: str, reason_prefix: str) -> None:
    route = route_query(query, known_projects=("hermes-llmwiki-rag", "codequest"))
    assert route.retrieve is retrieve, route
    assert route.profile == profile, route
    assert route.reason.startswith(reason_prefix), route


def test_known_project_mention_routes_to_project_profile() -> None:
    route = route_query("what is left to do on codequest?", known_projects=("codequest",))
    assert route.profile == "project:codequest" and route.retrieve
    unknown = route_query("what is left to do on project:nope?", known_projects=("codequest",))
    assert unknown.profile == "answer"


def test_informative_terms_counts_non_stopwords() -> None:
    assert informative_terms("what is the sqlite-vec limit") == 2
    assert informative_terms("the of and") == 0


def _cand(i: int, *, dense=1, lex=1, rrf=0.05, dist=0.4, bm25=12.0, auth=True) -> Candidate:
    return Candidate(
        chunk_id=i,
        document_id=i,
        path=f"wiki/{i}.md",
        title="t",
        heading_path=("t",),
        section_name="",
        position=0,
        text="x",
        text_hash="h",
        source_kind="wiki",
        page_role="durable",
        project_id=None,
        updated_at_ns=1,
        is_route_map=False,
        dense_rank=dense,
        dense_distance=dist,
        lexical_rank=lex,
        bm25_score=bm25,
        rrf_score=rrf,
        authority_class="durable",
        authority_match=auth,
    )


def _result(cands) -> RetrievalResult:
    return RetrievalResult(query="q", profile="answer", mode="hybrid", candidates=tuple(cands))


def test_result_features_are_total_and_bounded() -> None:
    empty = result_features(_result([]))
    assert set(empty) == set(FEATURE_NAMES)
    assert all(v == 0.0 for v in empty.values())
    good = result_features(_result([_cand(1, rrf=0.09), _cand(2, rrf=0.03, dense=None, lex=2)]))
    assert good["top1_in_both_channels"] == 1.0
    assert good["rrf_margin"] == pytest.approx(0.06)
    assert 0.0 <= good["dense_top_similarity"] <= 1.0
    assert 0.0 < good["bm25_top_norm"] < 1.0
    assert good["channel_agreement_top5"] == pytest.approx(0.2)


def test_decide_without_gate_never_injects() -> None:
    d = decide(_result([_cand(1)]), None)
    assert d.inject is False and d.reason == "no-calibrated-gate"


def test_fit_gate_separates_clear_classes_and_threshold_hits_precision() -> None:
    positives = [
        ({**result_features(_result([_cand(1, rrf=0.08 + i * 0.001)]))}, True) for i in range(20)
    ]
    negatives = [
        (
            {
                **result_features(
                    _result([_cand(1, dense=None, lex=1, rrf=0.02, dist=1.2, bm25=2.0, auth=False)])
                )
            },
            False,
        )
        for _ in range(20)
    ]
    samples = positives + negatives
    gate = fit_gate(samples, epochs=800)
    scores_pos = [gate.score(f) for f, _ in positives]
    scores_neg = [gate.score(f) for f, _ in negatives]
    assert min(scores_pos) > max(scores_neg)
    threshold = choose_threshold(gate, samples, min_precision=0.9)
    certified = InjectionGate(gate.weights, gate.bias, threshold)
    report = gate_report(certified, [(f, label, not label) for f, label in samples])
    assert (
        report["precision"] == 1.0 and report["abstain_rate"] == 1.0 and report["coverage"] == 1.0
    )
    assert report["pollution"] == 0.0


def test_gate_roundtrip_through_json(tmp_path: Path) -> None:
    gate = InjectionGate(
        weights={"rrf_top": 1.5},
        bias=-0.2,
        threshold=0.6,
        fitted_on="v1@abc",
        metrics={"gate_a_passed": True},
    )
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate.to_dict()))
    loaded = load_gate(path)
    assert loaded == gate
    assert load_gate(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("{not json")
    assert load_gate(tmp_path / "bad.json") is None
    d = decide(_result([_cand(1, rrf=0.9)]), loaded)
    assert d.inject is True and d.reason == "above-threshold"
    assert decide(_result([]), loaded).reason == "no-candidates"
