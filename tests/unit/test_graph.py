"""Wikilink normalisation, transactional link projection, resolution, and expansion."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmwiki import db as dbmod
from llmwiki.graph import (
    graph_summary,
    neighbours,
    normalise_target,
    project_scope_document_ids,
    replace_document_links,
    resolve_links,
)


@pytest.mark.parametrize(
    ("raw", "key", "hint"),
    [
        ("Note", "note", None),
        ("Note|alias", "note", None),
        ("Note#Heading|alias", "note", None),
        ("folder/Note.md", "note", "folder/note"),
        ("../architecture", "architecture", "../architecture"),
        ("projects/relationships", "relationships", "projects/relationships"),
        ("", "", None),
        ("#only-heading", "", None),
    ],
)
def test_normalise_target(raw: str, key: str, hint: str | None) -> None:
    assert normalise_target(raw) == (key, hint)


def _doc(
    conn,
    path: str,
    *,
    project_id=None,
    page_role="durable",
    source_kind="wiki",
    aliases=(),
    links=(),
) -> int:
    import json

    cursor = conn.execute(
        "INSERT INTO documents (path, absolute_path, title, mtime_ns, size_bytes, content_hash, "
        "indexed_at_ns, source_kind, page_role, project_id, aliases_json, wikilinks_json) "
        "VALUES (?, ?, ?, 1, 1, ?, 1, ?, ?, ?, ?, ?)",
        (
            path,
            f"/v/{path}",
            path,
            path,
            source_kind,
            page_role,
            project_id,
            json.dumps(list(aliases)),
            json.dumps(list(links)),
        ),
    )
    doc_id = int(cursor.lastrowid)
    replace_document_links(conn, doc_id, path=path, aliases=aliases, wikilinks=links)
    return doc_id


@pytest.fixture
def conn(tmp_path: Path):
    with dbmod.connect(tmp_path / "g.sqlite") as c:
        dbmod.init_schema(c)
        yield c


def test_resolution_by_stem_alias_path_and_relative_path(conn) -> None:
    arch = _doc(conn, "wiki/projects/rag/architecture.md", project_id="rag", page_role="project")
    other_arch = _doc(
        conn, "wiki/projects/other/architecture.md", project_id="other", page_role="project"
    )
    concept = _doc(conn, "wiki/reciprocal-rank-fusion.md", aliases=("RRF",))
    research = _doc(
        conn,
        "wiki/projects/rag/research/index.md",
        project_id="rag",
        page_role="route-map",
        links=(
            "../architecture",
            "RRF",
            "projects/other/architecture",
            "missing-page",
            "wiki/reciprocal-rank-fusion",
        ),
    )
    assert resolve_links(conn) == 4
    rows = dict(
        conn.execute(
            "SELECT target_text, target_document_id FROM links WHERE source_document_id = ?",
            (research,),
        ).fetchall()
    )
    assert rows["../architecture"] == arch  # relative path wins over the other project's page
    assert rows["RRF"] == concept  # alias
    assert rows["projects/other/architecture"] == other_arch  # trailing path fragment
    assert rows["wiki/reciprocal-rank-fusion"] == concept
    assert rows["missing-page"] is None
    summary = graph_summary(conn)
    assert summary == {"links": 5, "resolved": 4, "unresolved": 1}
    # Idempotent: nothing left to resolve.
    assert resolve_links(conn) == 0


def test_ambiguous_stem_prefers_same_folder_then_same_project(conn) -> None:
    a_plan = _doc(conn, "wiki/projects/a/plan.md", project_id="a", page_role="project")
    b_plan = _doc(conn, "wiki/projects/b/plan.md", project_id="b", page_role="project")
    from_b = _doc(
        conn, "wiki/projects/b/brief.md", project_id="b", page_role="project", links=("plan",)
    )
    from_root = _doc(conn, "wiki/overview.md", links=("plan",))
    resolve_links(conn)
    targets = {
        int(s): t
        for s, t in conn.execute(
            "SELECT source_document_id, target_document_id FROM links"
        ).fetchall()
    }
    assert targets[from_b] == b_plan
    assert targets[from_root] == a_plan  # lowest id when no folder/project cue


def test_links_replaced_atomically_and_cascade_on_delete(conn) -> None:
    target = _doc(conn, "wiki/t.md")
    src = _doc(conn, "wiki/s.md", links=("t",))
    resolve_links(conn)
    assert graph_summary(conn)["resolved"] == 1
    with pytest.raises(RuntimeError), dbmod.transaction(conn):
        replace_document_links(conn, src, path="wiki/s.md", aliases=(), wikilinks=("t", "u"))
        assert graph_summary(conn)["links"] == 2
        raise RuntimeError("boom")
    assert graph_summary(conn)["links"] == 1  # rolled back with the transaction
    conn.execute("DELETE FROM documents WHERE id = ?", (target,))
    row = conn.execute(
        "SELECT target_document_id FROM links WHERE source_document_id = ?", (src,)
    ).fetchone()
    assert row[0] is None  # ON DELETE SET NULL keeps provenance of the dangling link
    conn.execute("DELETE FROM documents WHERE id = ?", (src,))
    assert graph_summary(conn)["links"] == 0


def test_neighbours_bounded_and_cycle_safe(conn) -> None:
    a = _doc(conn, "wiki/a.md", links=("b",))
    b = _doc(conn, "wiki/b.md", links=("c", "a"))
    c = _doc(conn, "wiki/c.md", links=("a",))
    d = _doc(conn, "wiki/d.md", links=("a",))  # backlink only
    resolve_links(conn)
    assert neighbours(conn, [a], hops=1, direction="out") == {b}
    assert neighbours(conn, [a], hops=1, direction="in") == {b, c, d}
    assert neighbours(conn, [a], hops=2, direction="out") == {b, c}
    assert neighbours(conn, [a], hops=5, direction="both", max_nodes=2).issubset({b, c, d})
    assert len(neighbours(conn, [a], hops=5, direction="both", max_nodes=2)) == 2
    assert neighbours(conn, [a], hops=0) == set()


def test_project_scope_admits_only_curated_linked_pages(conn) -> None:
    cs = _doc(
        conn,
        "wiki/projects/rag/current-state.md",
        project_id="rag",
        page_role="current-state",
        links=("sqlite-vec", "log", "index", "rrf-paper", "idea"),
    )
    _idx = _doc(
        conn,
        "wiki/projects/rag/index.md",
        project_id="rag",
        page_role="route-map",
        links=("current-state",),
    )
    _log = _doc(conn, "wiki/projects/rag/log.md", project_id="rag", page_role="log")
    vec = _doc(conn, "wiki/sqlite-vec.md", links=("fastembed",))
    fe = _doc(conn, "wiki/fastembed.md")
    _paper = _doc(conn, "raw/papers/rrf-paper.md", source_kind="raw", page_role="evidence")
    _idea = _doc(conn, "Clippings/ideas/idea.md", source_kind="clipping", page_role="idea")
    resolve_links(conn)
    scope = project_scope_document_ids(conn, "rag", hops=1)
    assert scope == {cs, vec}  # workspace page + linked curated page; no log/index/raw/idea
    two_hops = project_scope_document_ids(conn, "rag", hops=2)
    assert two_hops == {cs, vec, fe}
    assert project_scope_document_ids(conn, "nope") == set()
