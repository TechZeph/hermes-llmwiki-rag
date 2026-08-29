"""Unit tests for the markdown parser (Phase 1).

These tests are pure: no filesystem, no database. They construct
small markdown strings in :func:`_write` and assert on the parsed
view.
"""

from __future__ import annotations

from pathlib import Path

from llmwiki.parser import parse_markdown


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_frontmatter_yaml_is_parsed(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "note.md",
        """---
title: My Note
tags: [foo, bar]
aliases: [First Alias, Second]
---

# Heading

Body.
""",
    )
    parsed = parse_markdown(str(p))
    assert parsed.title == "My Note"
    assert parsed.tags == ("foo", "bar")
    assert parsed.aliases == ("First Alias", "Second")


def test_first_h1_is_title_when_no_frontmatter_title(tmp_path: Path) -> None:
    p = _write(tmp_path, "x.md", "# My Heading\n\nbody\n")
    parsed = parse_markdown(str(p))
    assert parsed.title == "My Heading"


def test_filename_is_title_when_no_h1_and_no_frontmatter_title(tmp_path: Path) -> None:
    p = _write(tmp_path, "fallback-name.md", "no heading here, just prose.\n")
    parsed = parse_markdown(str(p))
    # When there's no H1 and no frontmatter title, the parser returns "" and
    # the indexer falls back to the filename. This test asserts the parser's
    # contract: empty string.
    assert parsed.title == ""


def test_wikilinks_are_extracted(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "x.md",
        """Some text with [[Note A]] and [[Note B#Heading]] and [[Note C|alias]].

Duplicate [[Note A]] should be deduplicated.
""",
    )
    parsed = parse_markdown(str(p))
    assert parsed.wikilinks == ("Note A", "Note B#Heading", "Note C|alias")


def test_tags_inside_code_fences_are_ignored(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "x.md",
        """Real tag: #real

```
This is inside a fence: #not-a-tag
```

Another real one: #also-real
""",
    )
    parsed = parse_markdown(str(p))
    assert "real" in parsed.tags
    assert "also-real" in parsed.tags
    assert "not-a-tag" not in parsed.tags


def test_heading_outline_is_collected(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "x.md",
        """# H1
## H2 first
text
### H3
## H2 second
""",
    )
    parsed = parse_markdown(str(p))
    assert [h["text"] for h in parsed.headings] == ["H1", "H2 first", "H3", "H2 second"]
    assert [h["level"] for h in parsed.headings] == [1, 2, 3, 2]


def test_malformed_frontmatter_does_not_crash(tmp_path: Path) -> None:
    """Obsidian occasionally has half-broken frontmatter. The parser must not raise."""
    p = _write(
        tmp_path,
        "x.md",
        """---
tags: [unclosed
title: still works
---

# Title still works
""",
    )
    parsed = parse_markdown(str(p))
    # We don't assert exact content here; the contract is "no exception".
    assert parsed.title in ("still works", "Title still works", "")


def test_aliases_dedup_and_strip(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "x.md",
        """---
aliases: [ " Alpha ", "Beta", "Alpha", "" ]
---

body
""",
    )
    parsed = parse_markdown(str(p))
    assert "Alpha" in parsed.aliases
    assert "Beta" in parsed.aliases
    # Empty string filtered out.
    assert "" not in parsed.aliases
    # Duplicates collapsed.
    assert list(parsed.aliases).count("Alpha") == 1


def test_frontmatter_only_singular_alias(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "x.md",
        """---
alias: Single
---

body
""",
    )
    parsed = parse_markdown(str(p))
    assert parsed.aliases == ("Single",)
