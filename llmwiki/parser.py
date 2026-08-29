"""Markdown + frontmatter parsing.

Phase 1 only needs:

- The YAML frontmatter (Obsidian properties block).
- The first H1 as the document title (or the frontmatter ``title`` if present).
- The set of ``[[wikilinks]]`` anywhere in the body.
- The set of ``#tags`` (Obsidian-style, in the body, not in code fences).
- The heading outline (levels 1-6).
- The set of ``|alias`` declarations in the frontmatter (Obsidian's
  ``aliases: [foo, bar]`` or ``alias: foo`` style).

We deliberately do *not* parse the body into a tree in Phase 1.
The structural chunker (Phase 2) will use ``markdown-it-py`` for that.
For Phase 1, regex on the raw text is honest, fast, and easy to test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

import frontmatter
import yaml

# --- regex patterns ----------------------------------------------------------

# [[Wiki Link]] or [[Wiki Link|alias]] or [[path#anchor|alias]] or [[path#anchor]]
_WIKILINK_RE: Final = re.compile(r"\[\[([^\[\]\n]+?)\]\]")

# #tag (not at start of line inside a code fence; we approximate by skipping
# blocks first). Tags cannot contain whitespace and must start with a letter
# or digit. Obsidian allows nested tags like #parent/child.
_TAG_RE: Final = re.compile(r"(?:^|\s)#([A-Za-z0-9_\-][A-Za-z0-9_/\-]*)")

# ATX headings: #, ##, ..., ###### at the start of a line.
_HEADING_RE: Final = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)

# ```...``` fenced code blocks. We strip them before scanning for tags /
# wikilinks so we don't pick up #tag inside a code example.
_FENCE_RE: Final = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)


# --- data shape --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """The structured view of one markdown file."""

    frontmatter: dict[str, Any]
    title: str
    tags: tuple[str, ...]
    wikilinks: tuple[str, ...]
    aliases: tuple[str, ...]
    headings: tuple[dict[str, Any], ...]


# --- implementation ----------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text)


def _coerce_frontmatter(raw: object) -> dict[str, Any]:
    """Normalise frontmatter into a plain dict.

    ``python-frontmatter`` returns whatever the YAML parser produced.
    YAML can produce tuples for inline lists, and a top-level
    non-dict (e.g. just a string) is possible for malformed input.
    We defensively coerce.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    # Best-effort: try to load as YAML once more in case the lib
    # already parsed it into something exotic.
    try:
        parsed = yaml.safe_load(raw)  # type: ignore[arg-type]
    except yaml.YAMLError:
        return {"_raw": str(raw)}
    if isinstance(parsed, dict):
        return parsed
    return {"_value": parsed}


def _normalise_tags(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = [str(x) for x in raw]
    else:
        items = [str(raw)]
    # Strip leading "#" if present, lowercase, dedupe, preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        normalised = item.lstrip("#").strip().lower()
        if normalised and normalised not in seen:
            seen.add(normalised)
            out.append(normalised)
    return tuple(out)


def _normalise_aliases(fm: dict[str, Any]) -> tuple[str, ...]:
    aliases: list[str] = []
    for key in ("aliases", "alias"):
        value = fm.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            aliases.append(value)
        elif isinstance(value, list):
            aliases.extend(str(x) for x in value)
    # Dedupe, preserve order, strip whitespace.
    seen: set[str] = set()
    out: list[str] = []
    for a in aliases:
        a = a.strip()
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return tuple(out)


def _extract_wikilinks(body: str) -> tuple[str, ...]:
    """Return the unique wikilink targets in document order.

    The target string is the raw inner content, e.g.
    ``"Note Title"``, ``"folder/Note"``, ``"Note#Heading"``,
    ``"Note|alias"``. The pipe-separated alias is *not* stripped:
    callers that want only the bare target can split on ``|`` themselves.
    """
    seen: set[str] = set()
    out: list[str] = []
    for match in _WIKILINK_RE.finditer(body):
        target = match.group(1).strip()
        if target and target not in seen:
            seen.add(target)
            out.append(target)
    return tuple(out)


def _extract_tags(body: str) -> tuple[str, ...]:
    """Return unique tag names found in the body, lowercase, no leading ``#``."""
    cleaned = _strip_code_fences(body)
    seen: set[str] = set()
    out: list[str] = []
    for match in _TAG_RE.finditer(cleaned):
        name = match.group(1).lower()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def _extract_headings(body: str) -> tuple[dict[str, Any], ...]:
    """Return the document's heading outline (in document order)."""
    cleaned = _strip_code_fences(body)
    return tuple(
        {"level": len(match.group(1)), "text": match.group(2).strip()}
        for match in _HEADING_RE.finditer(cleaned)
    )


def _first_h1(body: str) -> str | None:
    cleaned = _strip_code_fences(body)
    for match in _HEADING_RE.finditer(cleaned):
        if len(match.group(1)) == 1:
            return match.group(2).strip()
    return None


def parse_markdown(path: str) -> ParsedDocument:
    """Read a markdown file and return its structured view.

    Uses :mod:`python-frontmatter` for the frontmatter (it handles
    Obsidian's properties block correctly) and the regex helpers
    above for body extraction. The file is read once; the function
    is pure.

    Malformed YAML frontmatter is tolerated: the frontmatter block is
    treated as absent and the rest of the file is still parsed. The
    raw bad text is not preserved; if you need it, parse the file
    manually.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            post = frontmatter.load(fh)
        fm = _coerce_frontmatter(post.metadata)
    except yaml.YAMLError:
        # Malformed frontmatter; treat as no frontmatter.
        with open(path, encoding="utf-8") as fh:
            body = fh.read()
        # Strip a leading frontmatter block if one exists so the rest
        # of the body still parses cleanly.
        if body.startswith("---"):
            end = body.find("\n---", 3)
            if end != -1:
                body = body[body.find("\n", end) + 1 :].lstrip("\n")
        fm = {}
    else:
        body = post.content

    title = (
        str(fm.get("title")).strip()
        if isinstance(fm.get("title"), str) and fm.get("title")
        else (_first_h1(body) or "")
    )

    fm_tags = _normalise_tags(fm.get("tags"))
    body_tags = _extract_tags(body)
    merged_tags: list[str] = []
    seen: set[str] = set()
    for tag in (*fm_tags, *body_tags):
        if tag not in seen:
            seen.add(tag)
            merged_tags.append(tag)

    return ParsedDocument(
        frontmatter=fm,
        title=title,
        tags=tuple(merged_tags),
        aliases=_normalise_aliases(fm),
        wikilinks=_extract_wikilinks(body),
        headings=_extract_headings(body),
    )


def all_tags(parsed: ParsedDocument) -> tuple[str, ...]:
    """Return tags from both frontmatter and body, deduplicated, in order.

    Frontmatter tags come first, then body tags. This is the union
    stored in the ``documents.tags_json`` column.
    """
    seen: set[str] = set()
    out: list[str] = []
    for source in (parsed.tags, _extract_tags_after_strip(parsed)):
        for tag in source:
            if tag not in seen:
                seen.add(tag)
                out.append(tag)
    return tuple(out)


def _extract_tags_after_strip(parsed: ParsedDocument) -> tuple[str, ...]:
    """Helper: re-extract body tags when we only have the parsed view."""
    # The regex helpers operate on raw text; for an already-parsed
    # document we don't have the raw text, so this is currently a
    # no-op. Body tags are extracted in :func:`parse_markdown` and
    # would be added to the ParsedDocument in Phase 2 if needed.
    return ()


__all__ = ["ParsedDocument", "all_tags", "parse_markdown"]
