"""Lightweight entity graph and communities (V3, no LLM).

In a curated wiki the durable *pages* are the entities: every page has a
title and optional aliases, and other pages mention them in prose without
always linking. This module projects two structures from that:

- ``mentions``: chunk → document edges where the chunk text contains a
  page's title or alias as a phrase (FTS5 phrase match, so tokenisation
  matches retrieval). Together with resolved wikilinks this gives a
  mention graph that supports "what refers to X" and one-hop expansion
  without entity extraction models.
- ``communities``: a deterministic label-propagation partition of the
  undirected wikilink graph between non-hub pages, with stable tie-breaks
  so a rebuild produces the same labels. Used for "related pages" and
  for grouping status output; never as a ranking signal.

Both are rebuilt at the end of an index run inside one transaction; both
are cheap (hundreds of phrase queries, a few passes over ~3k edges).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Final

from .lexical import FTS_TABLE, build_match_query

MIN_KEY_CHARS: Final = 5
MAX_MENTIONS_PER_KEY: Final = 400
# A title that appears in more chunks than this is a generic phrase, not an
# entity mention (e.g. "LLM Wiki" in a wiki about LLM wikis); skip it.
MAX_MENTION_DOCUMENT_FREQUENCY: Final = 40
_GENERIC_KEYS: Final = frozenset(
    {
        "index",
        "log",
        "plan",
        "brief",
        "context",
        "questions",
        "decisions",
        "current-state",
        "next-actions",
        "architecture",
        "readme",
        "notes",
        "todo",
        "summary",
        "overview",
    }
)


def _mention_keys(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    """``(document_id, phrase)`` pairs worth searching for: titles and aliases."""
    out: list[tuple[int, str]] = []
    rows = conn.execute(
        "SELECT d.id, d.title, k.key FROM documents d JOIN link_keys k ON k.document_id = d.id "
        "WHERE d.source_kind = 'wiki' AND d.page_role NOT IN ('log', 'route-map')"
    ).fetchall()
    seen: set[tuple[int, str]] = set()
    for doc_id, title, key in rows:
        for phrase in (str(title), str(key).replace("-", " ")):
            phrase = phrase.strip()
            if len(phrase) < MIN_KEY_CHARS or phrase.lower() in _GENERIC_KEYS:
                continue
            terms = build_match_query(phrase).count('"') // 2
            if terms == 0 or (terms == 1 and len(phrase) < 8):
                continue
            item = (int(doc_id), phrase.lower())
            if item in seen:
                continue
            seen.add(item)
            out.append((int(doc_id), phrase))
    return out


def rebuild_mentions(conn: sqlite3.Connection) -> int:
    """Recompute ``mentions`` from scratch; returns the number of edges."""
    conn.execute("DELETE FROM mentions")
    inserted = 0
    for doc_id, phrase in _mention_keys(conn):
        quoted = '"' + phrase.replace('"', "") + '"'
        rows = conn.execute(
            f"SELECT f.rowid, c.document_id FROM {FTS_TABLE} f JOIN chunks c ON c.id = f.rowid "
            f"WHERE {FTS_TABLE} MATCH ? AND c.document_id != ? LIMIT ?",
            (quoted, doc_id, MAX_MENTIONS_PER_KEY),
        ).fetchall()
        if not rows:
            continue
        if len({int(r[1]) for r in rows}) > MAX_MENTION_DOCUMENT_FREQUENCY:
            continue  # generic phrase, not a distinguishing entity
        conn.executemany(
            "INSERT OR IGNORE INTO mentions(chunk_id, document_id) VALUES (?, ?)",
            [(int(r[0]), doc_id) for r in rows],
        )
        inserted += len(rows)
    return inserted


def _undirected_edges(conn: sqlite3.Connection) -> dict[int, set[int]]:
    """Community graph: resolved links between non-hub pages.

    Route maps (indexes) link to everything and would collapse the graph into
    one community, so they are excluded; mentions are used for related-page
    ranking only, because common phrases behave like hubs too.
    """
    adj: dict[int, set[int]] = {}
    hubs = {
        int(r[0])
        for r in conn.execute(
            "SELECT id FROM documents WHERE is_route_map = 1 OR page_role = 'route-map'"
        ).fetchall()
    }
    for a, b in conn.execute(
        "SELECT source_document_id, target_document_id FROM links WHERE target_document_id IS NOT NULL"
    ).fetchall():
        a, b = int(a), int(b)
        if a == b or a in hubs or b in hubs:
            continue
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    for (doc_id,) in conn.execute("SELECT id FROM documents").fetchall():
        adj.setdefault(int(doc_id), set())
    return adj


def label_propagation(adj: dict[int, set[int]], *, max_iterations: int = 20) -> dict[int, int]:
    """Deterministic label propagation: nodes visited in id order, ties to the lowest label."""
    labels = {node: node for node in adj}
    for _ in range(max_iterations):
        changed = False
        for node in sorted(adj):
            counts: dict[int, int] = {}
            for nb in adj[node]:
                counts[labels[nb]] = counts.get(labels[nb], 0) + 1
            if not counts:
                continue
            best = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if best != labels[node]:
                labels[node] = best
                changed = True
        if not changed:
            break
    # Normalise labels to the lowest member id of each community.
    members: dict[int, list[int]] = {}
    for node, label in labels.items():
        members.setdefault(label, []).append(node)
    normalised: dict[int, int] = {}
    for group in members.values():
        root = min(group)
        for node in group:
            normalised[node] = root
    return normalised


def rebuild_communities(conn: sqlite3.Connection) -> int:
    """Recompute ``communities`` from links + mentions; returns the community count."""
    conn.execute("DELETE FROM communities")
    adj = _undirected_edges(conn)
    labels = label_propagation(adj)
    conn.executemany(
        "INSERT INTO communities(document_id, community_id) VALUES (?, ?)",
        [(node, label) for node, label in labels.items()],
    )
    return len(set(labels.values()))


def refresh_entity_graph(conn: sqlite3.Connection) -> dict[str, int]:
    edges = rebuild_mentions(conn)
    communities = rebuild_communities(conn)
    return {"mention_edges": edges, "communities": communities}


@dataclass(frozen=True, slots=True)
class RelatedPage:
    path: str
    title: str
    relation: str  # links-to | linked-from | mentions | mentioned-in | same-community
    weight: int


def related_pages(conn: sqlite3.Connection, path: str, *, limit: int = 20) -> list[RelatedPage]:
    """Pages related to ``path`` via links, mentions, and community membership."""
    row = conn.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()
    if row is None:
        return []
    doc_id = int(row[0])
    scores: dict[int, dict[str, int]] = {}

    def add(other: int, relation: str, weight: int = 1) -> None:
        if other == doc_id:
            return
        scores.setdefault(other, {})[relation] = (
            scores.setdefault(other, {}).get(relation, 0) + weight
        )

    for (t,) in conn.execute(
        "SELECT target_document_id FROM links WHERE source_document_id = ? AND target_document_id IS NOT NULL",
        (doc_id,),
    ):
        add(int(t), "links-to", 3)
    for (s,) in conn.execute(
        "SELECT source_document_id FROM links WHERE target_document_id = ?", (doc_id,)
    ):
        add(int(s), "linked-from", 2)
    for (t,) in conn.execute(
        "SELECT DISTINCT m.document_id FROM mentions m JOIN chunks c ON c.id = m.chunk_id WHERE c.document_id = ?",
        (doc_id,),
    ):
        add(int(t), "mentions", 2)
    for (s,) in conn.execute(
        "SELECT DISTINCT c.document_id FROM mentions m JOIN chunks c ON c.id = m.chunk_id WHERE m.document_id = ?",
        (doc_id,),
    ):
        add(int(s), "mentioned-in", 1)
    community = conn.execute(
        "SELECT community_id FROM communities WHERE document_id = ?", (doc_id,)
    ).fetchone()
    if community is not None:
        for (member,) in conn.execute(
            "SELECT document_id FROM communities WHERE community_id = ? ORDER BY document_id LIMIT 200",
            (int(community[0]),),
        ):
            add(int(member), "same-community", 1)
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))[:limit]
    ids = [d for d, _ in ranked]
    placeholders = ",".join("?" * len(ids))
    meta = {
        int(r[0]): (str(r[1]), str(r[2]))
        for r in conn.execute(
            f"SELECT id, path, title FROM documents WHERE id IN ({placeholders})", ids
        )
    }
    out: list[RelatedPage] = []
    for other, relations in ranked:
        if other not in meta:
            continue
        top = max(relations.items(), key=lambda kv: (kv[1], kv[0]))[0]
        out.append(RelatedPage(meta[other][0], meta[other][1], top, sum(relations.values())))
    return out


def community_summary(conn: sqlite3.Connection, *, limit: int = 10) -> list[dict[str, object]]:
    rows = conn.execute(
        "SELECT community_id, COUNT(*) AS n FROM communities GROUP BY community_id ORDER BY n DESC, community_id LIMIT ?",
        (limit,),
    ).fetchall()
    out: list[dict[str, object]] = []
    for community_id, n in rows:
        titles = [
            str(r[0])
            for r in conn.execute(
                "SELECT d.title FROM communities c JOIN documents d ON d.id = c.document_id "
                "WHERE c.community_id = ? ORDER BY d.id LIMIT 5",
                (int(community_id),),
            ).fetchall()
        ]
        out.append({"community_id": int(community_id), "size": int(n), "sample_titles": titles})
    return out


def graph_counts(conn: sqlite3.Connection) -> dict[str, int]:
    mentions = int(conn.execute("SELECT COUNT(*) FROM mentions").fetchone()[0])
    communities = int(
        conn.execute("SELECT COUNT(DISTINCT community_id) FROM communities").fetchone()[0]
    )
    return {"mention_edges": mentions, "communities": communities}


__all__ = [
    "RelatedPage",
    "community_summary",
    "graph_counts",
    "label_propagation",
    "rebuild_communities",
    "rebuild_mentions",
    "refresh_entity_graph",
    "related_pages",
]
