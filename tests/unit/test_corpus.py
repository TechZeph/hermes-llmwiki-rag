"""Tests for deterministic corpus classification and profile membership."""

from __future__ import annotations

from pathlib import Path

from llmwiki import db as dbmod
from llmwiki.corpus import classify_path, filter_candidate_ids, profile_matches


def test_classify_path_derives_authority_metadata_from_vault_layout() -> None:
    """Canonical vault paths map to stable source and page roles."""
    assert classify_path("wiki/current-topic.md") == {
        "source_kind": "wiki",
        "page_role": "durable",
        "project_id": None,
        "is_route_map": False,
    }
    assert classify_path("wiki/projects/hosp-core/current-state.md") == {
        "source_kind": "wiki",
        "page_role": "current-state",
        "project_id": "hosp-core",
        "is_route_map": False,
    }
    assert classify_path("wiki/projects/hosp-core/decisions.md") == {
        "source_kind": "wiki",
        "page_role": "decision",
        "project_id": "hosp-core",
        "is_route_map": False,
    }
    assert classify_path("wiki/index-projects.md") == {
        "source_kind": "wiki",
        "page_role": "route-map",
        "project_id": None,
        "is_route_map": True,
    }
    assert classify_path("wiki/log.md") == {
        "source_kind": "wiki",
        "page_role": "log",
        "project_id": None,
        "is_route_map": False,
    }
    assert classify_path("raw/papers/retrieval.pdf.md") == {
        "source_kind": "raw",
        "page_role": "evidence",
        "project_id": None,
        "is_route_map": False,
    }
    assert classify_path("Clippings/ideas/rough-note.md") == {
        "source_kind": "clipping",
        "page_role": "idea",
        "project_id": None,
        "is_route_map": False,
    }


def test_profile_membership_keeps_default_answers_curated() -> None:
    """The default answer profile excludes logs, sources, and operational files."""
    assert profile_matches("answer", classify_path("wiki/current-topic.md"))
    assert not profile_matches("answer", classify_path("wiki/log.md"))
    assert not profile_matches("answer", classify_path("raw/papers/retrieval.pdf.md"))
    assert not profile_matches("answer", classify_path("Clippings/ideas/rough-note.md"))
    assert profile_matches("evidence", classify_path("raw/papers/retrieval.pdf.md"))
    assert profile_matches("history", classify_path("wiki/log.md"))
    assert profile_matches("all", classify_path("AGENTS.md"))
    assert profile_matches(
        "project:hosp-core", classify_path("wiki/projects/hosp-core/current-state.md")
    )
    assert not profile_matches(
        "project:hosp-core", classify_path("wiki/projects/urban-farm/current-state.md")
    )


def test_filter_candidate_ids_preserves_similarity_order_within_profile(tmp_path: Path) -> None:
    """A profile removes ineligible vector candidates without reranking the rest."""
    db_path = tmp_path / "llmwiki.sqlite"
    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        document_ids: dict[str, int] = {}
        for path in ("raw/source.md", "wiki/log.md", "wiki/durable.md"):
            metadata = classify_path(path)
            document_ids[path] = int(
                conn.execute(
                    """
                    INSERT INTO documents (
                        path, absolute_path, title, mtime_ns, size_bytes, content_hash, indexed_at_ns,
                        source_kind, page_role, project_id, updated_at_ns, is_route_map
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        path,
                        f"/vault/{path}",
                        path,
                        1,
                        1,
                        path,
                        1,
                        metadata["source_kind"],
                        metadata["page_role"],
                        metadata["project_id"],
                        1,
                        int(metadata["is_route_map"]),
                    ),
                ).lastrowid
            )
        chunk_ids = [
            int(
                conn.execute(
                    "INSERT INTO chunks (document_id, position, text, text_hash, char_count, indexed_at_ns) "
                    "VALUES (?, 0, 'body', ?, 4, 1)",
                    (document_ids[path], path),
                ).lastrowid
            )
            for path in ("raw/source.md", "wiki/log.md", "wiki/durable.md")
        ]

        assert filter_candidate_ids(conn, chunk_ids, profile="answer") == [chunk_ids[2]]
        assert filter_candidate_ids(conn, chunk_ids, profile="evidence") == [chunk_ids[0]]
        assert filter_candidate_ids(conn, chunk_ids, profile="history") == [chunk_ids[1]]
