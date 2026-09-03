"""Watcher coalescing and a real filesystem round-trip."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from watchdog.events import FileClosedEvent, FileModifiedEvent, FileOpenedEvent

from llmwiki import db as dbmod
from llmwiki.config import Settings
from llmwiki.watch import Coalescer, _is_indexable_event, watch_vault
from tests.helpers import SAMPLE_KEYWORDS, SAMPLE_VAULT, KeywordEmbedder, write_vault


def test_coalescer_collapses_bursts_and_respects_debounce() -> None:
    c = Coalescer(debounce_s=1.0)
    assert not c.due(10.0)
    c.note("wiki/a.md", 10.0)
    c.note("wiki/b.md", 10.5)
    assert not c.due(11.0)  # quiet for only 0.5s
    assert c.due(11.5)
    assert c.drain() == {"wiki/a.md", "wiki/b.md"}
    assert not c.due(20.0)


def test_watcher_ignores_read_only_events() -> None:
    path = "/vault/wiki/page.md"
    assert not _is_indexable_event(FileOpenedEvent(path))
    assert not _is_indexable_event(FileClosedEvent(path))
    assert _is_indexable_event(FileModifiedEvent(path))


def test_watch_runs_on_create_modify_and_delete(tmp_path: Path) -> None:
    vault = write_vault(tmp_path / "vault", SAMPLE_VAULT)
    settings = Settings(vault_path=vault, db_path=tmp_path / "db.sqlite")
    embedder = KeywordEmbedder(SAMPLE_KEYWORDS)
    stop = threading.Event()
    seen: list[tuple[int, int, int, set[str]]] = []
    done = threading.Event()

    def on_run(stats, paths):
        seen.append(
            (stats.documents_added, stats.documents_updated, stats.documents_removed, paths)
        )
        if len(seen) >= 3:
            done.set()

    thread = threading.Thread(
        target=lambda: watch_vault(
            settings,
            embedder=embedder,
            debounce_s=0.3,
            poll_s=0.05,
            stop_event=stop,
            on_run=on_run,
            initial_run=True,
        ),
        daemon=True,
    )
    thread.start()
    # Wait for the startup run.
    deadline = time.time() + 15
    while len(seen) < 1 and time.time() < deadline:
        time.sleep(0.05)
    assert seen and seen[0][0] > 0, "startup run should index the vault"

    new_page = vault / "wiki" / "new-page.md"
    new_page.write_text("# New page\n\nfastembed arena notes\n", encoding="utf-8")
    while len(seen) < 2 and time.time() < deadline:
        time.sleep(0.05)
    assert len(seen) >= 2 and seen[1][0] == 1 and "wiki/new-page.md" in seen[1][3]

    new_page.unlink()
    (vault / ".obsidian" / "workspace.json").write_text("{}", encoding="utf-8")  # ignored
    while len(seen) < 3 and time.time() < deadline:
        time.sleep(0.05)
    stop.set()
    thread.join(timeout=10)
    assert len(seen) >= 3 and seen[2][2] == 1
    with dbmod.connect(settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM documents WHERE path = 'wiki/new-page.md'"
            ).fetchone()[0]
            == 0
        )
    report = dbmod.inspect_integrity(settings.db_path, vault_path=vault)
    assert report["ok"] is True
