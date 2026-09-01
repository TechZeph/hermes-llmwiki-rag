"""File watching with event coalescing (V2).

``watch_vault`` observes the vault with ``watchdog`` and runs one
incremental index pass after the filesystem has been quiet for
``debounce_s`` seconds. Bursts of events (an editor saving several
files, a sync client landing a batch) collapse into a single run, and a
run never overlaps another. Deleted or moved files are handled by the
incremental indexer's normal "missing document" path, so the watcher
carries no state that could go stale across restarts.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .embeddings import Embedder
from .indexer import Indexer, _should_skip
from .logging import get_logger
from .models import IndexRunStats

logger = get_logger("watch")


@dataclass(slots=True)
class Coalescer:
    """Pure event coalescer: remembers that something changed and when."""

    debounce_s: float = 2.0
    pending: bool = False
    last_event_at: float = 0.0
    changed_paths: set[str] = field(default_factory=set)

    def note(self, rel_path: str, now: float) -> None:
        self.pending = True
        self.last_event_at = now
        if len(self.changed_paths) < 1000:
            self.changed_paths.add(rel_path)

    def due(self, now: float) -> bool:
        return self.pending and (now - self.last_event_at) >= self.debounce_s

    def drain(self) -> set[str]:
        paths = set(self.changed_paths)
        self.changed_paths.clear()
        self.pending = False
        return paths


def _relevant(vault: Path, settings: Settings, raw_path: str) -> str | None:
    try:
        rel = Path(raw_path).resolve().relative_to(vault.resolve()).as_posix()
    except (OSError, ValueError):
        return None
    if not rel.endswith(".md"):
        return None
    if _should_skip(rel, settings.ignored_dirs, settings.ignored_globs):
        return None
    return rel


def watch_vault(
    settings: Settings,
    *,
    embedder: Embedder | None,
    debounce_s: float = 2.0,
    poll_s: float = 0.25,
    stop_event: threading.Event | None = None,
    on_run: Callable[[IndexRunStats, set[str]], None] | None = None,
    max_runs: int | None = None,
    initial_run: bool = True,
) -> int:
    """Block watching ``settings.vault_path``; returns the number of index runs performed.

    ``stop_event`` ends the loop; ``max_runs`` ends it after that many runs
    (tests). ``initial_run`` performs one incremental pass before watching
    so changes made while the watcher was down are not missed.
    """
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    vault = settings.vault_path
    indexer = Indexer(settings, embedder=embedder)
    coalescer = Coalescer(debounce_s=debounce_s)
    lock = threading.Lock()
    stop = stop_event or threading.Event()
    runs = 0

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event: FileSystemEvent) -> None:
            if event.is_directory:
                return
            for candidate in (getattr(event, "src_path", None), getattr(event, "dest_path", None)):
                if not candidate:
                    continue
                rel = _relevant(vault, settings, str(candidate))
                if rel is None:
                    continue
                with lock:
                    coalescer.note(rel, time.monotonic())

    def run_once(reason: str, paths: set[str]) -> None:
        nonlocal runs
        stats = indexer.run(mode="incremental")
        runs += 1
        logger.info(
            "watch run (%s): changed=%d added=%d updated=%d removed=%d errors=%d",
            reason,
            len(paths),
            stats.documents_added,
            stats.documents_updated,
            stats.documents_removed,
            len(stats.errors),
        )
        if on_run is not None:
            on_run(stats, paths)

    if initial_run:
        run_once("startup", set())
        if max_runs is not None and runs >= max_runs:
            return runs

    observer = Observer()
    observer.schedule(Handler(), str(vault), recursive=True)
    observer.daemon = True
    observer.start()
    logger.info("watching %s (debounce %.1fs)", vault.name, debounce_s)
    try:
        while not stop.is_set():
            time.sleep(poll_s)
            with lock:
                ready = coalescer.due(time.monotonic())
                paths = coalescer.drain() if ready else set()
            if ready:
                try:
                    run_once("change", paths)
                except Exception as exc:  # keep watching; the next run retries
                    logger.warning("watch run failed: %s", type(exc).__name__)
                if max_runs is not None and runs >= max_runs:
                    break
    finally:
        observer.stop()
        observer.join(timeout=5)
    return runs


__all__ = ["Coalescer", "watch_vault"]
