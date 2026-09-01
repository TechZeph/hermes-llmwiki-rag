"""Intent detection, authority ordering, and conflict labelling."""

from __future__ import annotations

from llmwiki.authority import (
    apply_authority_policy,
    authority_class,
    detect_conflicts,
    detect_intent,
)
from llmwiki.models import Candidate


def _cand(
    chunk_id: int,
    *,
    path: str,
    page_role: str,
    source_kind: str = "wiki",
    project_id=None,
    updated=1,
    route_map=False,
) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        document_id=chunk_id,
        path=path,
        title=path,
        heading_path=(path,),
        section_name="",
        position=0,
        text="t",
        text_hash="h",
        source_kind=source_kind,
        page_role=page_role,
        project_id=project_id,
        updated_at_ns=updated,
        is_route_map=route_map,
    )


def test_detect_intent_rules_are_ordered_and_deterministic() -> None:
    assert detect_intent("What is the current status of the RAG project?") == "current-state"
    assert detect_intent("Why did we choose sqlite-vec over faiss?") == "decision"
    assert detect_intent("When was sqlite-vec first added?") == "chronology"
    assert detect_intent("What does the RRF paper say about Condorcet?") == "evidence"
    assert detect_intent("How does reciprocal rank fusion work?") == "general"
    # Chronology cue wins over a current-state cue in the same query.
    assert detect_intent("When did the current status change?") == "chronology"


def test_authority_class_mapping() -> None:
    assert (
        authority_class(_cand(1, path="raw/p.md", page_role="evidence", source_kind="raw"))
        == "evidence"
    )
    assert (
        authority_class(
            _cand(1, path="Clippings/ideas/i.md", page_role="idea", source_kind="clipping")
        )
        == "idea"
    )
    assert (
        authority_class(_cand(1, path="wiki/index.md", page_role="route-map", route_map=True))
        == "route-map"
    )
    assert (
        authority_class(
            _cand(1, path="wiki/projects/x/current-state.md", page_role="current-state")
        )
        == "current-state"
    )
    assert (
        authority_class(_cand(1, path="wiki/projects/x/brief.md", page_role="project")) == "project"
    )
    assert authority_class(_cand(1, path="wiki/a.md", page_role="durable")) == "durable"


def test_current_state_intent_promotes_within_window_and_demotes_route_maps() -> None:
    cands = [
        _cand(1, path="wiki/index.md", page_role="route-map", route_map=True),
        _cand(2, path="wiki/a.md", page_role="durable"),
        _cand(
            3, path="wiki/projects/x/current-state.md", page_role="current-state", project_id="x"
        ),
        _cand(4, path="wiki/b.md", page_role="durable"),
    ]
    ordered, _ = apply_authority_policy(cands, intent="current-state", profile="answer")
    assert [c.chunk_id for c in ordered] == [3, 2, 4, 1]
    assert ordered[0].authority_match is True
    assert ordered[0].authority_class == "current-state"
    assert ordered[-1].authority_class == "route-map"


def test_promotion_is_bounded_by_window() -> None:
    cands = [_cand(i, path=f"wiki/{i}.md", page_role="durable") for i in range(1, 6)]
    cands.append(
        _cand(
            99, path="wiki/projects/x/current-state.md", page_role="current-state", project_id="x"
        )
    )
    ordered, _ = apply_authority_policy(cands, intent="current-state", profile="answer", window=3)
    assert [c.chunk_id for c in ordered] == [1, 2, 3, 4, 5, 99]


def test_general_intent_keeps_channel_order_among_curated_pages() -> None:
    cands = [
        _cand(1, path="wiki/a.md", page_role="durable"),
        _cand(2, path="wiki/projects/x/decisions.md", page_role="decision", project_id="x"),
        _cand(3, path="wiki/log.md", page_role="log"),
        _cand(4, path="wiki/b.md", page_role="durable"),
    ]
    ordered, _ = apply_authority_policy(cands, intent="general", profile="answer")
    assert [c.chunk_id for c in ordered] == [1, 2, 4, 3]


def test_evidence_profile_only_labels() -> None:
    cands = [
        _cand(1, path="raw/a.md", page_role="evidence", source_kind="raw"),
        _cand(2, path="Clippings/ideas/i.md", page_role="idea", source_kind="clipping"),
    ]
    ordered, _ = apply_authority_policy(cands, intent="general", profile="evidence")
    assert [c.chunk_id for c in ordered] == [1, 2]
    assert ordered[1].authority_class == "idea"


def test_conflict_labels() -> None:
    cands = [
        _cand(
            1,
            path="wiki/projects/x/current-state.md",
            page_role="current-state",
            project_id="x",
            updated=200,
        ),
        _cand(2, path="wiki/projects/x/brief.md", page_role="project", project_id="x", updated=100),
        _cand(3, path="wiki/projects/y/plan.md", page_role="project", project_id="y", updated=150),
    ]
    labels = detect_conflicts(cands)
    assert "competing-projects: x, y" in labels
    assert any(
        label.startswith("older-than-current-state: wiki/projects/x/brief.md") for label in labels
    )
    assert detect_conflicts(cands[:1]) == ()
