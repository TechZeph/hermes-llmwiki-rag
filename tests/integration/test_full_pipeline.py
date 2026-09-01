"""Integration tests for the RAG core (Phase 1).

These tests exercise the full pipeline against a real (small) vault
on disk, including the CLI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from llmwiki import db as dbmod
from llmwiki.config import Settings
from llmwiki.indexer import Indexer


def _make_vault(root: Path) -> Path:
    (root / "real-a.md").write_text(
        """---
title: Real A
tags: [a-tag]
---

# Real A

body with [[Real B]] link and a #body-tag.
""",
        encoding="utf-8",
    )
    (root / "sub").mkdir()
    (root / "sub" / "real-b.md").write_text(
        """---
title: Real B
aliases: [B-Alias]
---

# Real B

body.
""",
        encoding="utf-8",
    )
    (root / ".obsidian").mkdir()
    (root / ".obsidian" / "workspace.json").write_text("{}", encoding="utf-8")
    (root / ".hidden.md").write_text("# Hidden", encoding="utf-8")
    return root


def test_full_pipeline_against_real_vault(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    _make_vault(vault)
    db = tmp_path / "llmwiki.sqlite"

    indexer = Indexer(Settings(vault_path=vault, db_path=db, file_watch=False, log_level="WARNING"))
    stats = indexer.run()

    assert stats.documents_added == 2
    assert stats.documents_removed == 0

    with dbmod.connect(db) as conn:
        rows = conn.execute(
            "SELECT path, title, tags_json, wikilinks_json, aliases_json, headings_json "
            "FROM documents ORDER BY path"
        ).fetchall()
    by_path = {r[0]: r for r in rows}

    assert set(by_path) == {"real-a.md", "sub/real-b.md"}

    a = by_path["real-a.md"]
    assert a[1] == "Real A"  # title
    # Tags are a union of frontmatter and body tags, in order.
    assert json.loads(a[2]) == ["a-tag", "body-tag"]
    assert json.loads(a[3]) == ["Real B"]  # wikilinks
    assert json.loads(a[4]) == []  # aliases
    assert json.loads(a[5]) == [{"level": 1, "text": "Real A"}]  # headings

    b = by_path["sub/real-b.md"]
    assert json.loads(b[4]) == ["B-Alias"]  # aliases
    assert json.loads(b[3]) == []  # no wikilinks


def test_cli_index_and_status(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    _make_vault(vault)
    db = tmp_path / "llmwiki.sqlite"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmwiki.cli",
            "--log-level",
            "WARNING",
            "index",
            "--vault",
            str(vault),
            "--db",
            str(db),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "added=2" in result.stdout

    status = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmwiki.cli",
            "--log-level",
            "WARNING",
            "status",
            "--db",
            str(db),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "documents: 2" in status.stdout
    assert "added=2" in status.stdout


def test_cli_status_json(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    _make_vault(vault)
    db = tmp_path / "llmwiki.sqlite"
    subprocess.run(
        [sys.executable, "-m", "llmwiki.cli", "index", "--vault", str(vault), "--db", str(db)],
        capture_output=True,
        text=True,
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmwiki.cli",
            "status",
            "--db",
            str(db),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["documents"] == 2
    assert payload["exists"] is True
    assert payload["last_run"]["added"] == 2


def test_cli_integrity_reports_clean_no_embed_projection(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    _make_vault(vault)
    db = tmp_path / "llmwiki.sqlite"
    Indexer(Settings(vault_path=vault, db_path=db)).run()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmwiki.cli",
            "integrity",
            "--vault",
            str(vault),
            "--db",
            str(db),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["orphan_vectors"] == 0
    assert payload["chunks_without_embeddings"] >= 1
