"""FTS5 lexical index: safe query construction and trigger-maintained sync."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmwiki import db as dbmod
from llmwiki.chunks import insert_chunks
from llmwiki.lexical import Fts5Index, build_match_query
from llmwiki.models import Chunk


def test_match_query_quotes_every_term_and_drops_stopwords() -> None:
    assert build_match_query("what is the sqlite-vec limit") == '"sqlite-vec" OR "limit"'


def test_match_query_neutralises_fts_operators() -> None:
    q = build_match_query('foo AND NOT bar* (baz:qux) "quoted"')
    assert q == '"foo" OR "bar" OR "baz" OR "qux" OR "quoted"'


def test_match_query_dedupes_case_insensitively_and_can_be_empty() -> None:
    assert build_match_query("Vec vec VEC") == '"Vec"'
    assert build_match_query("the of and") == ""
    assert build_match_query("") == ""


def _document(
    conn, path: str, title: str, *, source_kind="wiki", page_role="durable", project_id=None
) -> int:
    cursor = conn.execute(
        "INSERT INTO documents (path, absolute_path, title, mtime_ns, size_bytes, content_hash, "
        "indexed_at_ns, source_kind, page_role, project_id) VALUES (?, ?, ?, 1, 1, ?, 1, ?, ?, ?)",
        (path, f"/vault/{path}", title, path, source_kind, page_role, project_id),
    )
    return int(cursor.lastrowid)


def _chunk(document_id: int, position: int, text: str, heading: tuple[str, ...] = ("T",)) -> Chunk:
    return Chunk(
        id=None,
        document_id=document_id,
        heading_path=heading,
        section_name=heading[-1],
        text=text,
        position=position,
    )


@pytest.fixture
def conn(tmp_path: Path):
    with dbmod.connect(tmp_path / "lex.sqlite") as c:
        dbmod.init_schema(c)
        yield c


def test_insert_and_delete_keep_fts_in_sync(conn) -> None:
    doc = _document(conn, "wiki/a.md", "Alpha page")
    ids = insert_chunks(
        conn,
        doc,
        [_chunk(doc, 0, "sqlite-vec stores vectors"), _chunk(doc, 1, "unrelated farming text")],
    )
    index = Fts5Index(conn)
    assert index.count() == 2
    hits = index.search("sqlite-vec", 10)
    assert [cid for cid, _ in hits] == [ids[0]]
    assert hits[0][1] > 0
    conn.execute("DELETE FROM documents WHERE id = ?", (doc,))
    assert index.count() == 0


def test_rolled_back_chunk_replacement_rolls_back_fts_rows(conn) -> None:
    doc = _document(conn, "wiki/a.md", "Alpha page")
    insert_chunks(conn, doc, [_chunk(doc, 0, "original text about arena memory")])
    index = Fts5Index(conn)
    with pytest.raises(RuntimeError), dbmod.transaction(conn):
        insert_chunks(conn, doc, [_chunk(doc, 0, "replacement text about faiss")])
        assert index.search("faiss", 5)
        raise RuntimeError("boom")
    assert index.search("faiss", 5) == []
    assert len(index.search("arena", 5)) == 1


def test_title_and_heading_are_searchable_with_higher_weight(conn) -> None:
    doc_a = _document(conn, "wiki/a.md", "Zebra guide")
    doc_b = _document(conn, "wiki/b.md", "Other page")
    a_ids = insert_chunks(conn, doc_a, [_chunk(doc_a, 0, "body text mentions nothing special")])
    b_ids = insert_chunks(conn, doc_b, [_chunk(doc_b, 0, "the zebra appears in the body only")])
    hits = Fts5Index(conn).search("zebra", 10)
    assert {cid for cid, _ in hits} == {a_ids[0], b_ids[0]}
    assert hits[0][0] == a_ids[0]  # title match outranks body match


def test_profile_filter_applies_before_ranking(conn) -> None:
    wiki = _document(conn, "wiki/a.md", "Alpha")
    raw = _document(conn, "raw/papers/p.md", "Paper", source_kind="raw", page_role="evidence")
    log = _document(conn, "wiki/log.md", "Log", page_role="log")
    proj = _document(conn, "wiki/projects/x/brief.md", "Brief", page_role="project", project_id="x")
    ids = {}
    for name, doc in (("wiki", wiki), ("raw", raw), ("log", log), ("proj", proj)):
        ids[name] = insert_chunks(conn, doc, [_chunk(doc, 0, "shared token fusion appears here")])[
            0
        ]
    index = Fts5Index(conn)
    assert {c for c, _ in index.search("fusion", 10, profile="answer")} == {
        ids["wiki"],
        ids["proj"],
    }
    assert {c for c, _ in index.search("fusion", 10, profile="evidence")} == {ids["raw"]}
    assert {c for c, _ in index.search("fusion", 10, profile="history")} == {ids["log"]}
    assert {c for c, _ in index.search("fusion", 10, profile="project:x")} == {ids["proj"]}
    assert len(index.search("fusion", 10, profile="all")) == 4
    with pytest.raises(ValueError):
        index.search("fusion", 10, profile="bogus")


def test_rebuild_restores_projection_from_chunks(conn) -> None:
    doc = _document(conn, "wiki/a.md", "Alpha")
    insert_chunks(conn, doc, [_chunk(doc, 0, "alpha text"), _chunk(doc, 1, "beta text")])
    conn.execute("DELETE FROM chunks_fts")
    index = Fts5Index(conn)
    assert index.count() == 0
    assert index.rebuild() == 2
    assert len(index.search("beta", 5)) == 1


def test_integrity_reports_fts_drift(tmp_path: Path) -> None:
    db_path = tmp_path / "drift.sqlite"
    with dbmod.connect(db_path) as c:
        dbmod.init_schema(c)
        doc = _document(c, "wiki/a.md", "Alpha")
        insert_chunks(c, doc, [_chunk(doc, 0, "alpha text")])
        c.execute("DELETE FROM chunks_fts")
        c.execute(
            "INSERT INTO chunks_fts(rowid, text, section_name, heading, title) VALUES (999, 'x', '', '', '')"
        )
    report = dbmod.inspect_integrity(db_path)
    assert report["orphan_fts_rows"] == 1
    assert report["chunks_without_fts"] == 1
    assert report["ok"] is False
