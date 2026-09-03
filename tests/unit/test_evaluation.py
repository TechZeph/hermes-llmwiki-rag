"""Golden-set loading/validation and metric computation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmwiki.evaluation.golden import (
    GoldenSet,
    Question,
    RelevantSource,
    load_golden,
    merge_drafts,
    stratification_report,
    validate_golden,
)
from llmwiki.evaluation.metrics import (
    aggregate,
    candidate_matches,
    corpus_fingerprint,
    score_question,
)
from llmwiki.evaluation.runner import format_benchmark_markdown, format_comparison
from llmwiki.models import Candidate, RetrievalResult


def _cand(
    chunk_id: int,
    path: str,
    *,
    section: str = "",
    heading=("T",),
    page_role="durable",
    source_kind="wiki",
    klass="durable",
    doc=None,
) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        document_id=doc if doc is not None else chunk_id,
        path=path,
        title=path,
        heading_path=heading,
        section_name=section,
        position=0,
        text="t",
        text_hash="h",
        source_kind=source_kind,
        page_role=page_role,
        project_id=None,
        updated_at_ns=1,
        is_route_map=False,
        dense_rank=1,
        dense_distance=0.1,
        lexical_rank=None,
        bm25_score=None,
        rrf_score=0.5,
        authority_class=klass,
    )


def _result(cands, *, intent="general", conflicts=()) -> RetrievalResult:
    return RetrievalResult(
        query="q",
        profile="answer",
        mode="hybrid",
        candidates=tuple(cands),
        intent=intent,
        conflicts=tuple(conflicts),
        elapsed_ms=12.0,
    )


def _q(id="q1", *, relevant=(), mode="retrieve", klass="durable", category="concept") -> Question:
    return Question(
        id=id,
        category=category,
        split="dev",
        query="q",
        profile="answer",
        authority_class=klass,
        mode=mode,
        relevant=tuple(relevant),
    )


def test_candidate_matches_path_and_optional_sections() -> None:
    c = _cand(1, "wiki/a.md", section="Limits", heading=("A", "Limits"))
    assert candidate_matches(c, RelevantSource("wiki/a.md"))
    assert candidate_matches(c, RelevantSource("wiki/a.md", ("Limits",)))
    assert candidate_matches(c, RelevantSource("wiki/a.md", ("A",)))
    assert not candidate_matches(c, RelevantSource("wiki/a.md", ("Other",)))
    assert not candidate_matches(c, RelevantSource("wiki/b.md"))


def test_score_question_rank_metrics_dedupe_documents() -> None:
    cands = [
        _cand(1, "wiki/x.md"),
        _cand(2, "wiki/x.md", doc=1),
        _cand(3, "wiki/a.md", klass="durable"),
        _cand(4, "wiki/b.md"),
    ]
    q = _q(relevant=[RelevantSource("wiki/a.md"), RelevantSource("wiki/b.md")])
    out = score_question(q, _result(cands), citation_ok=[True, True, True, False])
    assert out.hit_ranks == [2, 3]  # x.md counted once
    assert out.first_hit_rank == 2
    assert out.reciprocal_rank == pytest.approx(0.5)
    assert out.hit_at["1"] == 0.0 and out.hit_at["3"] == 1.0
    assert out.recall_at["3"] == pytest.approx(1.0)
    assert out.recall_at["1"] == pytest.approx(0.0)
    assert 0 < out.ndcg_at_10 < 1
    assert out.authority_top1_match is True
    assert out.duplicate_concentration == pytest.approx(0.5)
    assert out.citation_ok_fraction == pytest.approx(0.75)


def test_section_pinned_relevance_can_be_satisfied_by_later_chunk_of_same_doc() -> None:
    cands = [
        _cand(1, "wiki/a.md", section="Intro", heading=("A", "Intro")),
        _cand(2, "wiki/a.md", section="Limits", heading=("A", "Limits"), doc=1),
    ]
    q = _q(relevant=[RelevantSource("wiki/a.md", ("Limits",))])
    out = score_question(q, _result(cands), citation_ok=[True, True])
    assert out.first_hit_rank == 1


def test_authority_match_uses_question_class() -> None:
    cands = [
        _cand(
            1, "wiki/projects/x/current-state.md", page_role="current-state", klass="current-state"
        )
    ]
    q = _q(
        relevant=[RelevantSource("wiki/projects/x/current-state.md")],
        klass="current-state",
        category="current-state",
    )
    assert score_question(q, _result(cands), citation_ok=[True]).authority_top1_match is True
    q2 = _q(relevant=[RelevantSource("wiki/projects/x/current-state.md")], klass="decision")
    assert score_question(q2, _result(cands), citation_ok=[True]).authority_top1_match is False


def test_abstain_question_records_features_but_no_authority() -> None:
    q = _q(mode="abstain", klass="none", category="no-answer")
    out = score_question(q, _result([_cand(1, "wiki/a.md")]), citation_ok=[True])
    assert out.authority_top1_match is None
    assert out.features["rrf_top"] == pytest.approx(0.5)
    agg = aggregate([out])
    assert agg.n_abstain == 1 and agg.n_retrieve == 0
    assert agg.abstain_mean_top_rrf == pytest.approx(0.5)
    assert agg.mrr is None


def test_aggregate_latency_percentiles_and_means() -> None:
    outs = []
    for i in range(4):
        q = _q(id=f"q{i}", relevant=[RelevantSource("wiki/a.md")])
        res = RetrievalResult(
            query="q",
            profile="answer",
            mode="hybrid",
            candidates=(_cand(1, "wiki/a.md"),),
            elapsed_ms=float(10 * (i + 1)),
        )
        outs.append(score_question(q, res, citation_ok=[True]))
    agg = aggregate(outs)
    assert agg.mrr == pytest.approx(1.0)
    assert agg.latency_p50_ms == pytest.approx(25.0)
    assert agg.latency_p95_ms == pytest.approx(38.5)
    table = format_comparison(
        [
            {
                "variant": "hybrid",
                "split": "dev",
                "overall": agg.to_dict(),
                "by_category": {},
                "peak_rss_mb": 1.0,
            }
        ]
    )
    assert "| hybrid | dev | 4 |" in table


def test_benchmark_report_labels_itself_as_contributor_reference() -> None:
    report = format_benchmark_markdown([])
    assert report.startswith(
        "# Benchmarks\n\n"
        "> Contributor and evaluation reference. These aggregate results explain the\n"
        "> defaults selected for version 0.1.0; they are not required for installation\n"
        "> or everyday use.\n\n"
        "Generated by `llmwiki eval report` from `evals/runs/`. "
        "Latest run per golden set, split and variant."
    )


def test_golden_validation_catches_schema_and_path_errors(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    (vault / "wiki").mkdir(parents=True)
    (vault / "wiki" / "a.md").write_text("# A\n\n## Real heading\n\ntext\n", encoding="utf-8")
    doc = {
        "corpus": "t",
        "version": "v0",
        "questions": [
            {
                "id": "q1",
                "category": "concept",
                "split": "dev",
                "query": "x",
                "profile": "answer",
                "authority_class": "durable",
                "mode": "retrieve",
                "relevant": [{"path": "wiki/a.md", "sections": ["Real heading"]}],
            },
            {
                "id": "q1",
                "category": "bogus",
                "split": "test",
                "query": "",
                "profile": "nope",
                "authority_class": "x",
                "mode": "retrieve",
                "relevant": [],
            },
            {
                "id": "q3",
                "category": "no-answer",
                "split": "heldout",
                "query": "y",
                "profile": "answer",
                "authority_class": "none",
                "mode": "abstain",
                "relevant": [{"path": "wiki/a.md"}],
            },
            {
                "id": "q4",
                "category": "concept",
                "split": "heldout",
                "query": "z",
                "profile": "answer",
                "authority_class": "durable",
                "mode": "retrieve",
                "relevant": [
                    {"path": "wiki/missing.md"},
                    {"path": "wiki/a.md", "sections": ["Fake"]},
                ],
            },
        ],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    golden = load_golden(path)
    problems = validate_golden(golden, vault=vault)
    joined = "\n".join(problems)
    for expected in (
        "duplicate id",
        "unknown category",
        "unknown split",
        "invalid profile",
        "empty query",
        "unknown authority_class",
        "abstain questions must have no relevant",
        "missing file wiki/missing.md",
        "heading 'Fake' not found",
    ):
        assert expected in joined, expected
    assert "q1: " in joined
    assert golden.subset(split="heldout").counts() == {
        "no-answer": {"dev": 0, "heldout": 1},
        "concept": {"dev": 0, "heldout": 1},
    }
    strat = stratification_report(golden, minimum_total=2)
    assert any("has no questions" in p for p in strat)


def test_merge_drafts_orders_by_category_then_id(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(
        json.dumps(
            {
                "questions": [
                    {"id": "na-1", "category": "no-answer"},
                    {"id": "cs-2", "category": "current-state"},
                ]
            }
        )
    )
    b.write_text(json.dumps({"questions": [{"id": "cs-1", "category": "current-state"}]}))
    merged = merge_drafts([a, b], corpus="c", version="v")
    assert [q["id"] for q in merged["questions"]] == ["cs-1", "cs-2", "na-1"]
    assert merged["corpus"] == "c"


def test_corpus_fingerprint_is_order_independent() -> None:
    assert corpus_fingerprint([("a", "1"), ("b", "2")]) == corpus_fingerprint(
        [("b", "2"), ("a", "1")]
    )
    assert corpus_fingerprint([("a", "1")]) != corpus_fingerprint([("a", "2")])


def test_real_golden_set_is_well_formed() -> None:
    # The real-vault set is a private development asset; skip in public checkouts.
    path = (
        Path(__file__).resolve().parents[2]
        / "private"
        / "evals"
        / "golden"
        / "clanker-vault-v1.json"
    )
    if not path.exists():
        pytest.skip("private golden set not present")
    golden: GoldenSet = load_golden(path)
    assert validate_golden(golden) == []
    assert stratification_report(golden) == []
    assert len(golden.questions) >= 60
