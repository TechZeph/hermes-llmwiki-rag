"""Page-mention edges, deterministic communities, and related-page ranking."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmwiki import db as dbmod
from llmwiki.config import Settings
from llmwiki.entities import (
    community_summary,
    graph_counts,
    label_propagation,
    refresh_entity_graph,
    related_pages,
)
from llmwiki.indexer import Indexer
from tests.helpers import SAMPLE_KEYWORDS, SAMPLE_VAULT, KeywordEmbedder, write_vault


def test_label_propagation_is_deterministic_and_normalised() -> None:
    adj = {1: {2}, 2: {1, 3}, 3: {2}, 10: {11}, 11: {10}, 20: set()}
    labels = label_propagation(adj)
    assert labels[1] == labels[2] == labels[3] == 1
    assert labels[10] == labels[11] == 10
    assert labels[20] == 20
    assert label_propagation(adj) == labels


@pytest.fixture(scope="module")
def projection(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("entities")
    vault = write_vault(root / "vault", SAMPLE_VAULT)
    settings = Settings(vault_path=vault, db_path=root / "db.sqlite")
    Indexer(settings, embedder=KeywordEmbedder(SAMPLE_KEYWORDS)).run()
    return settings


def test_index_run_builds_mentions_and_communities(projection) -> None:
    with dbmod.connect(projection.db_path) as conn:
        counts = graph_counts(conn)
        assert counts["mention_edges"] > 0 and counts["communities"] > 0
        # sqlite-vec is mentioned by name in several other pages.
        vec = conn.execute("SELECT id FROM documents WHERE path = 'wiki/sqlite-vec.md'").fetchone()[
            0
        ]
        mentioning = {
            str(r[0])
            for r in conn.execute(
                "SELECT DISTINCT d.path FROM mentions m JOIN chunks c ON c.id = m.chunk_id "
                "JOIN documents d ON d.id = c.document_id WHERE m.document_id = ?",
                (vec,),
            )
        }
        assert "wiki/projects/rag/decisions.md" in mentioning
        assert "wiki/sqlite-vec.md" not in mentioning  # never self
        # The sample vault links pages only through route maps (hubs), which are
        # excluded from the community graph, so every page is its own community.
        summary = community_summary(conn, limit=3)
        assert summary and all(c["size"] == 1 for c in summary)
        # Rebuilding produces identical labels.
        before = dict(conn.execute("SELECT document_id, community_id FROM communities").fetchall())
        with dbmod.transaction(conn):
            refresh_entity_graph(conn)
        after = dict(conn.execute("SELECT document_id, community_id FROM communities").fetchall())
        assert before == after


def test_related_pages_rank_links_and_mentions(projection) -> None:
    with dbmod.connect(projection.db_path) as conn:
        related = related_pages(conn, "wiki/sqlite-vec.md", limit=10)
        paths = [r.path for r in related]
        assert "wiki/projects/rag/decisions.md" in paths
        assert "wiki/index-tools.md" in paths  # links to sqlite-vec
        assert all(r.path != "wiki/sqlite-vec.md" for r in related)
        assert related == sorted(related, key=lambda r: -r.weight)
        assert related_pages(conn, "wiki/nope.md") == []


def test_service_related_payload(projection, tmp_path: Path) -> None:
    from llmwiki.service import ConfigError, ServiceConfig, WikiService

    svc = WikiService(
        ServiceConfig(vault=str(projection.vault_path), db=str(projection.db_path)),
        embedder_factory=lambda s: KeywordEmbedder(SAMPLE_KEYWORDS),
        gate_path=tmp_path / "no-gate.json",
    )
    payload = svc.related("/wiki/sqlite-vec.md")
    assert payload["found"] is True and payload["related"]
    assert svc.related("wiki/missing.md")["found"] is False
    with pytest.raises(ConfigError):
        svc.related("../etc/passwd")
    svc.close()


def test_communities_follow_links_between_non_hub_pages(tmp_path: Path) -> None:
    import json

    from llmwiki.graph import replace_document_links, resolve_links

    def doc(conn, path, *, page_role="durable", route_map=False, links=()):
        cur = conn.execute(
            "INSERT INTO documents (path, absolute_path, title, mtime_ns, size_bytes, content_hash, "
            "indexed_at_ns, source_kind, page_role, is_route_map, wikilinks_json) "
            "VALUES (?, ?, ?, 1, 1, ?, 1, 'wiki', ?, ?, ?)",
            (path, f"/v/{path}", path, path, page_role, int(route_map), json.dumps(list(links))),
        )
        doc_id = int(cur.lastrowid)
        replace_document_links(conn, doc_id, path=path, aliases=(), wikilinks=links)
        return doc_id

    with dbmod.connect(tmp_path / "c.sqlite") as conn:
        dbmod.init_schema(conn)
        a = doc(conn, "wiki/a.md", links=("b",))
        b = doc(conn, "wiki/b.md", links=("a",))
        c = doc(conn, "wiki/c.md", links=("d",))
        d = doc(conn, "wiki/d.md")
        lonely = doc(conn, "wiki/e.md")
        doc(conn, "wiki/index.md", page_role="route-map", route_map=True, links=("a", "c", "e"))
        resolve_links(conn)
        with dbmod.transaction(conn):
            stats = refresh_entity_graph(conn)
        labels = dict(conn.execute("SELECT document_id, community_id FROM communities").fetchall())
        assert labels[a] == labels[b] and labels[c] == labels[d]
        assert labels[a] != labels[c] and labels[lonely] not in (labels[a], labels[c])
        assert stats["communities"] >= 4
