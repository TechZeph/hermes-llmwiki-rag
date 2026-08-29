"""Unit tests for the structural chunker (Phase 2)."""

from __future__ import annotations

from pathlib import Path

from llmwiki.chunker import chunk_document, split_oversized
from llmwiki.parser import parse_markdown


def _parse(tmp_path: Path, name: str, content: str):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return parse_markdown(str(p))


# --- basic structural splits -----------------------------------------------


def test_no_headings_yields_single_chunk(tmp_path: Path) -> None:
    parsed = _parse(tmp_path, "x.md", "just prose, no headings at all.\n")
    chunks = chunk_document(parsed)
    assert len(chunks) == 1
    assert chunks[0].text == "just prose, no headings at all."
    assert chunks[0].heading_path == ()


def test_single_h1_yields_preamble_plus_section(tmp_path: Path) -> None:
    parsed = _parse(tmp_path, "x.md", "# Title\n\nbody under title.\n")
    chunks = chunk_document(parsed)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ("Title",)
    assert chunks[0].section_name == "Title"
    assert chunks[0].text == "body under title."


def test_two_h2s_yield_two_chunks(tmp_path: Path) -> None:
    parsed = _parse(
        tmp_path,
        "x.md",
        "# Title\n\n## A\n\nbody A\n\n## B\n\nbody B\n",
    )
    chunks = chunk_document(parsed)
    assert len(chunks) == 2
    assert [c.section_name for c in chunks] == ["A", "B"]
    assert [c.text for c in chunks] == ["body A", "body B"]
    assert [c.position for c in chunks] == [0, 1]


def test_nested_headings_get_correct_breadcrumb(tmp_path: Path) -> None:
    parsed = _parse(
        tmp_path,
        "x.md",
        "# Title\n\n## A\n\n### A.1\n\ndeep text\n",
    )
    chunks = chunk_document(parsed)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ("Title", "A", "A.1")
    assert chunks[0].section_name == "A.1"


def test_preamble_before_first_heading_is_preserved(tmp_path: Path) -> None:
    parsed = _parse(
        tmp_path,
        "x.md",
        "intro text before any heading\n\n# Title\n\nbody under title\n",
    )
    chunks = chunk_document(parsed)
    assert len(chunks) == 2
    assert chunks[0].section_name == ""
    assert chunks[0].text == "intro text before any heading"
    assert chunks[0].heading_path == ("Title",)  # preamble is filed under the title
    assert chunks[1].section_name == "Title"
    assert chunks[1].text == "body under title"


def test_h1_then_h2_then_h1_sibling_resets_stack(tmp_path: Path) -> None:
    """An H1 after an H2 closes the H2's section; the new H1 is its own level."""
    parsed = _parse(
        tmp_path,
        "x.md",
        "# T1\n\n## A\n\nA body\n\n# T2\n\nT2 body\n",
    )
    chunks = chunk_document(parsed)
    # Two chunks: T1>A and T2. T2 is at level 1, so T1 is not in its breadcrumb.
    assert [c.heading_path for c in chunks] == [("T1", "A"), ("T2",)]
    assert [c.section_name for c in chunks] == ["A", "T2"]


def test_document_with_only_frontmatter_yields_no_chunks(tmp_path: Path) -> None:
    parsed = _parse(
        tmp_path,
        "x.md",
        """---
title: Just Meta
---

""",
    )
    chunks = chunk_document(parsed)
    # Body is empty after the frontmatter, so no chunks.
    assert chunks == []


# --- position ordering ------------------------------------------------------


def test_positions_are_zero_indexed_and_contiguous(tmp_path: Path) -> None:
    parsed = _parse(
        tmp_path,
        "x.md",
        "# T\n\n## A\n\na\n\n## B\n\nb\n\n## C\n\nc\n",
    )
    chunks = chunk_document(parsed)
    assert [c.position for c in chunks] == [0, 1, 2]


# --- paragraph-boundary safety net -----------------------------------------


def test_oversized_section_is_split_at_paragraph_boundary(tmp_path: Path) -> None:
    body = (
        "# T\n\n## Big\n\n"
        + ("paragraph one. " * 5 + "\n\n")  # ~90 chars
        + ("paragraph two. " * 5 + "\n\n")  # ~90 chars
        + ("paragraph three. " * 5 + "\n")  # ~90 chars
    )
    parsed = _parse(tmp_path, "x.md", body)
    chunks = chunk_document(parsed, max_chunk_chars=200)
    # Each paragraph is ~90 chars; max_chunk is 200; first cut at
    # the boundary at ~90, then remainder fits in one chunk. So 2
    # chunks: one with paragraphs 1+2, one with paragraph 3.
    assert len(chunks) == 2
    # Joining the chunks reproduces the original text (modulo whitespace).
    assert "".join(c.text for c in chunks).replace("\n", "").replace(" ", "") == (
        "paragraphone." * 5 + "paragraphtwo." * 5 + "paragraphthree." * 5
    )
    # All chunks keep the same heading path and section name.
    assert {c.heading_path for c in chunks} == {("T", "Big")}
    assert {c.section_name for c in chunks} == {"Big"}


def test_split_oversized_returns_input_when_small_enough() -> None:
    assert split_oversized("hello world", 100) == ["hello world"]


def test_split_oversized_handles_empty() -> None:
    assert split_oversized("", 100) == []


def test_split_oversized_keeps_giant_paragraph_intact() -> None:
    """A single paragraph longer than max_chars is kept whole rather than fragmented."""
    giant = "x" * 5000
    out = split_oversized(giant, 100)
    assert out == [giant]


# --- shape contract ---------------------------------------------------------


def test_chunks_carry_document_id(tmp_path: Path) -> None:
    parsed = _parse(tmp_path, "x.md", "# T\n\nbody\n")
    chunks = chunk_document(parsed, document_id=42)
    assert chunks[0].document_id == 42


def test_section_name_is_last_segment_of_path(tmp_path: Path) -> None:
    parsed = _parse(
        tmp_path,
        "x.md",
        "# Doc\n\n## Section\n\nbody\n",
    )
    chunks = chunk_document(parsed)
    assert chunks[0].section_name == "Section"
    assert chunks[0].heading_path == ("Doc", "Section")
