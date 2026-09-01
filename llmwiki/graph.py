"""Resolved Obsidian wikilink graph (V2).

Every document's ``[[wikilinks]]`` are projected into the ``links`` table
inside the same transaction as its chunks, so the graph is never
inconsistent with the documents it describes. Targets are stored as raw
text plus a normalised key and resolved to document ids in a cheap pass
at the end of each index run (a link can point at a page indexed later).

Resolution mirrors Obsidian's rules closely enough for retrieval:

1. ``[[Note|alias]]`` and ``[[Note#Heading]]`` strip the alias/heading.
2. A target containing ``/`` is tried as a vault path relative to the
   source document's folder (``../`` allowed) and then from the vault
   root, with or without ``.md``.
3. Otherwise the target is matched by note name (file stem) or by a
   frontmatter alias, case-insensitively. Ambiguous stems prefer the
   same folder as the source, then the same project, then the lowest id.

Graph expansion is bounded (hop count and node cap), cycle-safe, and
provenance-preserving: expansion only widens the set of documents a
profile may draw from; ranking still comes from the retrieval channels.
"""

from __future__ import annotations

import posixpath
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LinkRow:
    source_document_id: int
    target_text: str
    target_key: str
    target_path_hint: str | None


def normalise_target(raw: str) -> tuple[str, str | None]:
    """Return ``(key, path_hint)`` for a raw wikilink target.

    ``key`` is the lower-cased note name used for stem/alias matching.
    ``path_hint`` is the lower-cased path (without ``.md``) when the link
    carries folder information, else ``None``.
    """
    target = raw.split("|", 1)[0].split("#", 1)[0].strip()
    if target.endswith(".md"):
        target = target[:-3]
    target = target.strip("/")
    if not target:
        return "", None
    key = posixpath.basename(target).lower()
    hint = target.lower() if "/" in target else None
    return key, hint


def link_rows_for_document(document_id: int, wikilinks: Iterable[str]) -> list[LinkRow]:
    rows: list[LinkRow] = []
    seen: set[str] = set()
    for raw in wikilinks:
        key, hint = normalise_target(raw)
        if not key or raw in seen:
            continue
        seen.add(raw)
        rows.append(LinkRow(document_id, raw, key, hint))
    return rows


def replace_document_links(
    conn: sqlite3.Connection,
    document_id: int,
    *,
    path: str,
    aliases: Sequence[str],
    wikilinks: Iterable[str],
) -> int:
    """Replace ``links`` and ``link_keys`` rows for one document. Returns link count."""
    conn.execute("DELETE FROM links WHERE source_document_id = ?", (document_id,))
    conn.execute("DELETE FROM link_keys WHERE document_id = ?", (document_id,))
    stem = posixpath.basename(path)
    stem = stem[:-3] if stem.endswith(".md") else stem
    keys = {stem.lower()}
    keys.update(a.strip().lower() for a in aliases if a and a.strip())
    conn.executemany(
        "INSERT OR IGNORE INTO link_keys(document_id, key) VALUES (?, ?)",
        [(document_id, k) for k in keys],
    )
    rows = link_rows_for_document(document_id, wikilinks)
    conn.executemany(
        "INSERT INTO links(source_document_id, target_text, target_key, target_path_hint) "
        "VALUES (?, ?, ?, ?)",
        [(r.source_document_id, r.target_text, r.target_key, r.target_path_hint) for r in rows],
    )
    return len(rows)


def _resolve_path_hint(hint: str, source_path: str, by_path: dict[str, int]) -> int | None:
    source_dir = posixpath.dirname(source_path)
    candidates = [posixpath.normpath(posixpath.join(source_dir, hint)).lower(), hint]
    for cand in candidates:
        cand = cand.strip("/")
        if cand in by_path:
            return by_path[cand]
    # Obsidian also accepts a trailing path fragment ("projects/other/architecture"
    # for "wiki/projects/other/architecture"); prefer the shortest matching path.
    suffix = "/" + hint.strip("/")
    matches = sorted((path for path in by_path if path.endswith(suffix)), key=len)
    if matches:
        return by_path[matches[0]]
    return None


def resolve_links(conn: sqlite3.Connection) -> int:
    """Resolve every unresolved link whose target now exists. Returns the resolved count."""
    docs = conn.execute("SELECT id, path, project_id FROM documents").fetchall()
    by_path: dict[str, int] = {}
    project_of: dict[int, str | None] = {}
    dir_of: dict[int, str] = {}
    for doc_id, path, project_id in docs:
        p = str(path)
        key_path = (p[:-3] if p.endswith(".md") else p).lower()
        by_path[key_path] = int(doc_id)
        project_of[int(doc_id)] = str(project_id) if project_id is not None else None
        dir_of[int(doc_id)] = posixpath.dirname(p).lower()
    keys: dict[str, list[int]] = {}
    for doc_id, key in conn.execute("SELECT document_id, key FROM link_keys").fetchall():
        keys.setdefault(str(key), []).append(int(doc_id))
    path_of = {int(d): str(p) for d, p, _ in docs}

    unresolved = conn.execute(
        "SELECT id, source_document_id, target_key, target_path_hint FROM links "
        "WHERE target_document_id IS NULL"
    ).fetchall()
    updates: list[tuple[int, int]] = []
    for link_id, source_id, target_key, hint in unresolved:
        source_id = int(source_id)
        target: int | None = None
        if hint:
            target = _resolve_path_hint(str(hint), path_of.get(source_id, ""), by_path)
        if target is None:
            options = [d for d in keys.get(str(target_key), []) if d != source_id]
            if options:
                src_dir = dir_of.get(source_id, "")
                src_project = project_of.get(source_id)

                def rank(
                    d: int, *, src_dir: str = src_dir, src_project: str | None = src_project
                ) -> tuple[int, int, int]:
                    return (
                        0 if dir_of.get(d) == src_dir else 1,
                        0 if src_project and project_of.get(d) == src_project else 1,
                        d,
                    )

                target = min(options, key=rank)
        if target is not None:
            updates.append((target, int(link_id)))
    if updates:
        conn.executemany("UPDATE links SET target_document_id = ? WHERE id = ?", updates)
    return len(updates)


def neighbours(
    conn: sqlite3.Connection,
    seed_ids: Iterable[int],
    *,
    hops: int = 1,
    max_nodes: int = 50,
    direction: str = "both",
) -> set[int]:
    """Document ids reachable from ``seed_ids`` within ``hops`` resolved links.

    Bounded by ``max_nodes`` (seeds excluded) and cycle-safe. ``direction``
    is ``out`` (links from seeds), ``in`` (backlinks to seeds), or ``both``.
    """
    seeds = {int(i) for i in seed_ids}
    if hops <= 0 or not seeds or max_nodes <= 0:
        return set()
    found: set[int] = set()
    frontier = set(seeds)
    for _ in range(hops):
        if not frontier or len(found) >= max_nodes:
            break
        placeholders = ",".join("?" * len(frontier))
        params = list(frontier)
        rows: list[tuple[int]] = []
        if direction in ("out", "both"):
            rows += conn.execute(
                f"SELECT DISTINCT target_document_id FROM links "
                f"WHERE source_document_id IN ({placeholders}) AND target_document_id IS NOT NULL "
                f"ORDER BY target_document_id",
                params,
            ).fetchall()
        if direction in ("in", "both"):
            rows += conn.execute(
                f"SELECT DISTINCT source_document_id FROM links "
                f"WHERE target_document_id IN ({placeholders}) ORDER BY source_document_id",
                params,
            ).fetchall()
        next_frontier: set[int] = set()
        for (doc_id,) in rows:
            d = int(doc_id)
            if d in seeds or d in found:
                continue
            if len(found) >= max_nodes:
                break
            found.add(d)
            next_frontier.add(d)
        frontier = next_frontier
    return found


def project_scope_document_ids(
    conn: sqlite3.Connection,
    project_id: str,
    *,
    hops: int = 1,
    max_linked: int = 40,
) -> set[int]:
    """Workspace pages plus curated wiki pages they link to (``project:<id>`` expansion).

    Only ``source_kind = 'wiki'`` pages that are neither logs nor route maps
    are admitted from expansion, so raw sources and navigation indexes never
    enter an answer through the graph.
    """
    workspace = {
        int(r[0])
        for r in conn.execute(
            "SELECT id FROM documents WHERE source_kind = 'wiki' AND project_id = ? "
            "AND page_role NOT IN ('log', 'route-map')",
            (project_id,),
        ).fetchall()
    }
    if not workspace:
        return set()
    linked = neighbours(conn, workspace, hops=hops, max_nodes=max_linked, direction="out")
    if linked:
        placeholders = ",".join("?" * len(linked))
        allowed = {
            int(r[0])
            for r in conn.execute(
                f"SELECT id FROM documents WHERE id IN ({placeholders}) AND source_kind = 'wiki' "
                f"AND page_role NOT IN ('log', 'route-map')",
                list(linked),
            ).fetchall()
        }
    else:
        allowed = set()
    return workspace | allowed


def graph_summary(conn: sqlite3.Connection) -> dict[str, int]:
    total = int(conn.execute("SELECT COUNT(*) FROM links").fetchone()[0])
    resolved = int(
        conn.execute("SELECT COUNT(*) FROM links WHERE target_document_id IS NOT NULL").fetchone()[
            0
        ]
    )
    return {"links": total, "resolved": resolved, "unresolved": total - resolved}


__all__ = [
    "LinkRow",
    "graph_summary",
    "link_rows_for_document",
    "neighbours",
    "normalise_target",
    "project_scope_document_ids",
    "replace_document_links",
    "resolve_links",
]
