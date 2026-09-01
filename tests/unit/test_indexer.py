"""Unit tests for the indexer (Phase 1).

These tests build small, synthetic vault directories on disk, point
the indexer at them, and assert on the resulting database state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmwiki import db as dbmod
from llmwiki.config import Settings
from llmwiki.indexer import Indexer, iter_vault_files, resolve_contained


def _vault(notes: dict[str, str]) -> Path:
    """Build a fake vault under a temp dir with the given {relpath: content}."""
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


def test_first_run_adds_every_document(tmp_path: Path) -> None:
    vault = _vault(
        {
            "a.md": "# A\n\nbody",
            "sub/b.md": "# B\n\nbody",
        }
    )
    db = tmp_path / "llmwiki.sqlite"

    indexer = Indexer(_settings(vault, db))
    stats = indexer.run(mode="incremental")

    assert stats.documents_seen == 2
    assert stats.documents_added == 2
    assert stats.documents_updated == 0
    assert stats.documents_skipped == 0
    assert stats.documents_removed == 0

    with dbmod.connect(db) as conn:
        rows = conn.execute("SELECT path, title FROM documents ORDER BY path").fetchall()
    assert rows == [("a.md", "A"), ("sub/b.md", "B")]


def test_second_run_with_no_changes_skips(tmp_path: Path) -> None:
    vault = _vault({"a.md": "# A\n\nbody"})
    db = tmp_path / "llmwiki.sqlite"

    Indexer(_settings(vault, db)).run()
    stats = Indexer(_settings(vault, db)).run()

    assert stats.documents_added == 0
    assert stats.documents_updated == 0
    assert stats.documents_skipped == 1


def test_modified_file_is_updated(tmp_path: Path) -> None:
    vault_path = _vault({"a.md": "# A\n\nfirst body"})
    db = tmp_path / "llmwiki.sqlite"

    Indexer(_settings(vault_path, db)).run()
    # Overwrite the file (new mtime + new content hash).
    (vault_path / "a.md").write_text("# A\n\nsecond body", encoding="utf-8")
    stats = Indexer(_settings(vault_path, db)).run()

    assert stats.documents_updated == 1
    assert stats.documents_skipped == 0
    with dbmod.connect(db) as conn:
        title, hash_ = conn.execute(
            "SELECT title, content_hash FROM documents WHERE path = 'a.md'"
        ).fetchone()
    assert title == "A"
    assert isinstance(hash_, str) and len(hash_) == 64


def test_deleted_file_is_removed(tmp_path: Path) -> None:
    vault_path = _vault({"a.md": "# A", "b.md": "# B"})
    db = tmp_path / "llmwiki.sqlite"

    Indexer(_settings(vault_path, db)).run()
    (vault_path / "a.md").unlink()
    stats = Indexer(_settings(vault_path, db)).run()

    assert stats.documents_removed == 1
    with dbmod.connect(db) as conn:
        paths = {row[0] for row in conn.execute("SELECT path FROM documents").fetchall()}
    assert paths == {"b.md"}


def test_ignored_dirs_and_hidden_files_are_skipped(tmp_path: Path) -> None:
    vault_path = _vault(
        {
            "real.md": "# Real",
            ".obsidian/workspace.json": "{}",  # should be skipped (matched by ignored_dirs)
            ".hidden.md": "# Hidden",  # should be skipped (matched by ignored_globs)
        }
    )
    db = tmp_path / "llmwiki.sqlite"

    stats = Indexer(_settings(vault_path, db)).run()

    assert stats.documents_seen == 1
    assert stats.documents_added == 1
    with dbmod.connect(db) as conn:
        paths = {row[0] for row in conn.execute("SELECT path FROM documents").fetchall()}
    assert paths == {"real.md"}


def test_frontmatter_tags_and_wikilinks_are_persisted(tmp_path: Path) -> None:
    vault_path = _vault(
        {
            "x.md": """---
title: Custom
tags: [alpha, beta]
---

# H1

Links to [[Other Note]] and [[Another|with alias]].
""",
        }
    )
    db = tmp_path / "llmwiki.sqlite"

    Indexer(_settings(vault_path, db)).run()
    with dbmod.connect(db) as conn:
        row = conn.execute(
            "SELECT title, tags_json, wikilinks_json FROM documents WHERE path='x.md'"
        ).fetchone()
    assert row is not None
    title, tags_json, wikilinks_json = row
    assert title == "Custom"
    import json

    assert json.loads(tags_json) == ["alpha", "beta"]
    assert json.loads(wikilinks_json) == ["Other Note", "Another|with alias"]


def test_index_run_is_logged(tmp_path: Path) -> None:
    vault = _vault({"a.md": "# A"})
    db = tmp_path / "llmwiki.sqlite"

    Indexer(_settings(vault, db)).run()
    with dbmod.connect(db) as conn:
        rows = conn.execute(
            "SELECT mode, documents_added, documents_skipped FROM index_runs"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "incremental"
    assert rows[0][1] == 1
    assert rows[0][2] == 0


def test_empty_vault_is_a_clean_run(tmp_path: Path) -> None:
    import tempfile

    vault = Path(tempfile.mkdtemp(prefix="llmwiki-empty-"))
    db = tmp_path / "llmwiki.sqlite"

    stats = Indexer(_settings(vault, db)).run()
    assert stats.documents_seen == 0
    with dbmod.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


def test_missing_vault_path_raises(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "does-not-exist", tmp_path / "db.sqlite")
    with pytest.raises(FileNotFoundError):
        Indexer(settings).run()


def test_resolve_contained_rejects_symlinks_even_when_target_is_inside_vault(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    target = vault / "target.md"
    target.write_text("# Target", encoding="utf-8")
    (vault / "linked.md").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        resolve_contained(vault, vault / "linked.md")


def test_indexer_skips_file_and_directory_symlinks(tmp_path: Path) -> None:
    vault = _vault({"safe.md": "# Safe"})
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("# Secret", encoding="utf-8")
    (vault / "file-link.md").symlink_to(outside / "secret.md")
    (vault / "directory-link").symlink_to(outside, target_is_directory=True)

    files = list(iter_vault_files(vault, _settings(vault, tmp_path / "db.sqlite")))

    assert [file.rel_path for file in files] == ["safe.md"]


def test_indexer_persists_path_derived_corpus_metadata(tmp_path: Path) -> None:
    """Indexing stores classification needed to filter retrieval by profile."""
    vault = _vault(
        {
            "wiki/current-topic.md": "# Current topic\n\nbody",
            "wiki/log.md": "# Log\n\nhistory",
            "wiki/projects/hosp-core/current-state.md": "# Current\n\nstate",
            "raw/papers/evidence.md": "# Evidence\n\nsource",
            "Clippings/ideas/rough-note.md": "# Rough note\n\nidea",
        }
    )
    db = tmp_path / "llmwiki.sqlite"

    Indexer(_settings(vault, db)).run()

    with dbmod.connect(db) as conn:
        rows = conn.execute(
            "SELECT path, source_kind, page_role, project_id, updated_at_ns, is_route_map "
            "FROM documents ORDER BY path"
        ).fetchall()

    assert [(path, kind, role, project_id, route_map) for path, kind, role, project_id, _, route_map in rows] == [
        ("Clippings/ideas/rough-note.md", "clipping", "idea", None, 0),
        ("raw/papers/evidence.md", "raw", "evidence", None, 0),
        ("wiki/current-topic.md", "wiki", "durable", None, 0),
        ("wiki/log.md", "wiki", "log", None, 0),
        ("wiki/projects/hosp-core/current-state.md", "wiki", "current-state", "hosp-core", 0),
    ]
    assert all(isinstance(updated_at_ns, int) and updated_at_ns > 0 for *_, updated_at_ns, _ in rows)
