"""Integration tests for the chunker inside the indexer (Phase 2)."""

from __future__ import annotations

import json
from pathlib import Path

from llmwiki import db as dbmod
from llmwiki.config import Settings
from llmwiki.indexer import Indexer


def _vault(notes: dict[str, str]) -> Path:
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="llmwiki-vault-"))
    for rel, content in notes.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def _settings(vault: Path, db: Path) -> Settings:
    return Settings(
        vault_path=vault,
        db_path=db,
        file_watch=False,
        ignored_dirs=(".obsidian", ".trash", ".git"),
        ignored_globs=("**/.*",),
        log_level="WARNING",
        log_format="text",
    )


def test_indexing_creates_chunks_with_correct_metadata(tmp_path: Path) -> None:
    vault = _vault(
        {
            "a.md": """# Title A

preamble text

## Section 1

body one

## Section 2

body two
""",
        }
    )
    db = tmp_path / "llmwiki.sqlite"

    stats = Indexer(_settings(vault, db)).run()
    assert stats.documents_added == 1
    # The document has 1 H1 + 2 H2s; no body-before-first-heading. So 3 chunks.
    assert stats.chunks_added == 3

    with dbmod.connect(db) as conn:
        rows = conn.execute(
            "SELECT position, section_name, heading_path_json, text FROM chunks ORDER BY position"
        ).fetchall()
    assert [r[0] for r in rows] == [0, 1, 2]
    # The H1 itself is a section like any other: section_name is "Title A",
    # body is "preamble text".
    assert [r[1] for r in rows] == ["Title A", "Section 1", "Section 2"]
    assert [json.loads(r[2]) for r in rows] == [
        ["Title A"],
        ["Title A", "Section 1"],
        ["Title A", "Section 2"],
    ]
    assert [r[3] for r in rows] == ["preamble text", "body one", "body two"]


def test_no_change_run_does_not_rechunk(tmp_path: Path) -> None:
    """Skipped documents should not touch their chunk rows."""
    vault = _vault({"a.md": "# T\n\nbody\n"})
    db = tmp_path / "llmwiki.sqlite"

    Indexer(_settings(vault, db)).run()
    stats = Indexer(_settings(vault, db)).run()
    assert stats.documents_skipped == 1
    assert stats.chunks_added == 0
    assert stats.chunks_updated == 0
    assert stats.chunks_removed == 0


def test_modified_document_replaces_chunks(tmp_path: Path) -> None:
    vault = _vault({"a.md": "# T\n\n## A\n\none\n\n## B\n\ntwo\n"})
    db = tmp_path / "llmwiki.sqlite"

    Indexer(_settings(vault, db)).run()
    (vault / "a.md").write_text(
        "# T\n\n## C\n\nthree\n",
        encoding="utf-8",
    )
    stats = Indexer(_settings(vault, db)).run()
    assert stats.documents_updated == 1
    # Old document had 2 chunks (A, B); new has 1 (C). removed=2, updated=1.
    assert stats.chunks_removed == 2
    assert stats.chunks_updated == 1

    with dbmod.connect(db) as conn:
        rows = conn.execute("SELECT section_name FROM chunks ORDER BY position").fetchall()
    assert [r[0] for r in rows] == ["C"]


def test_deleted_document_cascades_to_chunks(tmp_path: Path) -> None:
    vault = _vault({"a.md": "# T\n\nbody\n", "b.md": "# B\n\nbody\n"})
    db = tmp_path / "llmwiki.sqlite"

    Indexer(_settings(vault, db)).run()
    (vault / "a.md").unlink()
    Indexer(_settings(vault, db)).run()

    with dbmod.connect(db) as conn:
        chunk_paths = conn.execute(
            "SELECT d.path FROM chunks c JOIN documents d ON c.document_id = d.id"
        ).fetchall()
    assert {r[0] for r in chunk_paths} == {"b.md"}


def test_empty_vault_yields_no_chunks(tmp_path: Path) -> None:
    import tempfile

    vault = Path(tempfile.mkdtemp(prefix="llmwiki-empty-"))
    db = tmp_path / "llmwiki.sqlite"

    Indexer(_settings(vault, db)).run()
    with dbmod.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
