"""Deterministic corpus classification for vault-derived retrieval metadata."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import TypedDict


class CorpusMetadata(TypedDict):
    """Stable path-derived metadata persisted with every indexed document."""

    source_kind: str
    page_role: str
    project_id: str | None
    is_route_map: bool


def classify_path(path: str) -> CorpusMetadata:
    """Classify a vault-relative Markdown path without reading its content.

    Classification deliberately depends only on the vault's documented layout,
    so the same path always receives the same metadata after a rebuild.
    """
    parts = tuple(part for part in path.split("/") if part)
    filename = parts[-1] if parts else ""
    stem = filename.removesuffix(".md")

    if parts[:1] == ("raw",):
        return _metadata("raw", "evidence")
    if parts[:2] == ("Clippings", "ideas"):
        return _metadata("clipping", "idea")
    if parts[:1] == ("Clippings",):
        return _metadata("clipping", "evidence")
    if parts[:2] == ("wiki", "projects") and len(parts) >= 4:
        project_id = parts[2]
        return _metadata("wiki", _project_page_role(stem), project_id)
    if parts[:1] == ("wiki",):
        if path == "wiki/log.md":
            return _metadata("wiki", "log")
        if filename.startswith("index"):
            return _metadata("wiki", "route-map", is_route_map=True)
        return _metadata("wiki", "durable")
    return _metadata("operational", "operational")


def profile_matches(profile: str, metadata: Mapping[str, object]) -> bool:
    """Return whether path-derived metadata belongs to a retrieval profile.

    Graph expansion for linked canonical pages is intentionally not implied by
    ``project:<id>`` yet; resolved wikilinks are a later stage. This profile
    currently scopes directly to the selected project's workspace, minus its
    navigation index and append-only log (use ``history`` for logs).
    """
    source_kind = metadata["source_kind"]
    page_role = metadata["page_role"]
    project_id = metadata["project_id"]

    if profile == "answer":
        return source_kind == "wiki" and page_role not in {"log", "route-map"}
    if profile == "evidence":
        return source_kind in {"raw", "clipping"}
    if profile == "history":
        return source_kind == "wiki" and page_role == "log"
    if profile == "current":
        return source_kind == "wiki" and page_role == "current-state"
    if profile == "all":
        return True
    if profile.startswith("project:"):
        wanted_project = profile.removeprefix("project:")
        return (
            bool(wanted_project)
            and source_kind == "wiki"
            and project_id == wanted_project
            and page_role not in {"log", "route-map"}
        )
    raise ValueError("profile must be 'answer', 'evidence', 'history', 'all', or 'project:<id>'")


def profile_predicate(profile: str, *, alias: str = "d") -> tuple[str, list[object]]:
    """Return a SQL predicate over the ``documents`` table for ``profile``.

    Mirrors :func:`profile_matches` exactly so SQL-side filtering (FTS5,
    hydration) and Python-side filtering (vector over-fetch) agree.
    """
    a = alias
    if profile == "answer":
        return f"({a}.source_kind = 'wiki' AND {a}.page_role NOT IN ('log', 'route-map'))", []
    if profile == "evidence":
        return f"({a}.source_kind IN ('raw', 'clipping'))", []
    if profile == "history":
        return f"({a}.source_kind = 'wiki' AND {a}.page_role = 'log')", []
    if profile == "all":
        return "(1 = 1)", []
    if profile.startswith("project:"):
        wanted = profile.removeprefix("project:")
        if not wanted:
            raise ValueError("project profile requires an id: project:<id>")
        return (
            f"({a}.source_kind = 'wiki' AND {a}.project_id = ? "
            f"AND {a}.page_role NOT IN ('log', 'route-map'))",
            [wanted],
        )
    raise ValueError("profile must be 'answer', 'evidence', 'history', 'all', or 'project:<id>'")


def filter_candidate_ids(
    conn: sqlite3.Connection, candidate_ids: list[int], *, profile: str
) -> list[int]:
    """Keep vector candidates that belong to ``profile`` in their input order."""
    if not candidate_ids:
        return []
    placeholders = ",".join("?" * len(candidate_ids))
    rows = conn.execute(
        f"""
        SELECT c.id, d.source_kind, d.page_role, d.project_id, d.is_route_map
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.id IN ({placeholders})
        """,
        candidate_ids,
    ).fetchall()
    metadata_by_id: dict[int, CorpusMetadata] = {
        int(chunk_id): {
            "source_kind": str(source_kind),
            "page_role": str(page_role),
            "project_id": str(project_id) if project_id is not None else None,
            "is_route_map": bool(is_route_map),
        }
        for chunk_id, source_kind, page_role, project_id, is_route_map in rows
    }
    return [
        chunk_id
        for chunk_id in candidate_ids
        if (metadata := metadata_by_id.get(chunk_id)) is not None
        and profile_matches(profile, metadata)
    ]


def _project_page_role(stem: str) -> str:
    if stem == "current-state":
        return "current-state"
    if stem == "decisions":
        return "decision"
    if stem == "index":
        return "route-map"
    if stem == "log":
        return "log"
    return "project"


def _metadata(
    source_kind: str,
    page_role: str,
    project_id: str | None = None,
    *,
    is_route_map: bool = False,
) -> CorpusMetadata:
    return {
        "source_kind": source_kind,
        "page_role": page_role,
        "project_id": project_id,
        "is_route_map": is_route_map,
    }


__all__ = [
    "CorpusMetadata",
    "classify_path",
    "filter_candidate_ids",
    "profile_matches",
    "profile_predicate",
]
