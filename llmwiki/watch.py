"""File watching with event coalescing, plus the supervised in-host watcher (V2).

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
from typing import Any

from . import db as dbmod
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


COLD_START_MAX_MISSING = 200


@dataclass(slots=True)
class WatcherState:
    state: str = "idle"  # idle | starting | running | refused | stopped | failed
    reason: str = ""
    started_at: float | None = None
    runs: int = 0
    last_run_at: float | None = None
    last_run: dict[str, Any] = field(default_factory=dict)
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason": self.reason,
            "started_at": self.started_at,
            "runs": self.runs,
            "last_run_at": self.last_run_at,
            "last_run": dict(self.last_run),
            "last_error": self.last_error,
        }


def cold_start_check(settings: Settings) -> str:
    """Return an empty string when a watcher may start, else the refusal reason."""
    if not settings.db_path.exists():
        return "projection does not exist; run `llmwiki index --vault ...` once first"
    try:
        with dbmod.connect(settings.db_path) as conn:
            version_row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            if version_row is None:
                return "projection is uninitialised; run `llmwiki index` once first"
            missing = int(
                conn.execute(
                    "SELECT COUNT(*) FROM chunks c LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id "
                    "WHERE e.chunk_id IS NULL"
                ).fetchone()[0]
            )
    except Exception as exc:  # unreadable projection: do not start
        return f"projection unreadable ({type(exc).__name__}); run `llmwiki integrity`"
    if missing > COLD_START_MAX_MISSING:
        return (
            f"{missing} chunks lack vectors; a cold-start embed inside the gateway is refused, "
            "run `llmwiki index` once first"
        )
    return ""


class VaultWatcher:
    """Lifecycle wrapper around :func:`llmwiki.watch.watch_vault` for the plugin."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedder_factory: Callable[[Settings], Embedder],
        debounce_s: float = 2.0,
    ) -> None:
        self._settings = settings
        self._embedder_factory = embedder_factory
        self._debounce = debounce_s
        self._state = WatcherState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._state.to_dict()

    def start(self) -> bool:
        """Start once; returns True when the watcher is (now) running."""
        with self._lock:
            if self._state.state in ("running", "starting"):
                return True
            refusal = cold_start_check(self._settings)
            if refusal:
                self._state.state = "refused"
                self._state.reason = refusal
                return False
            self._state.state = "starting"
            self._state.reason = ""
            self._state.started_at = time.time()
            self._stop.clear()
            thread = threading.Thread(target=self._run, name="llmwiki-plugin-watcher", daemon=True)
            self._thread = thread
            thread.start()
            return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._lock:
            if self._state.state in ("running", "starting"):
                self._state.state = "stopped"

    def _on_run(self, stats: IndexRunStats, paths: set[str]) -> None:
        with self._lock:
            self._state.state = "running"
            self._state.runs += 1
            self._state.last_run_at = time.time()
            self._state.last_run = {
                "changed_paths": len(paths),
                "added": stats.documents_added,
                "updated": stats.documents_updated,
                "removed": stats.documents_removed,
                "embeddings_built": stats.embeddings_built,
                "errors": len(stats.errors),
            }

    def _run(self) -> None:
        try:
            embedder = self._embedder_factory(self._settings)
            with self._lock:
                self._state.state = "running"
            watch_vault(
                self._settings,
                embedder=embedder,
                debounce_s=self._debounce,
                stop_event=self._stop,
                on_run=self._on_run,
                initial_run=True,
            )
            with self._lock:
                if self._state.state != "failed":
                    self._state.state = "stopped"
        except Exception as exc:
            with self._lock:
                self._state.state = "failed"
                self._state.last_error = type(exc).__name__


__all__ = [
    "COLD_START_MAX_MISSING",
    "Coalescer",
    "VaultWatcher",
    "WatcherState",
    "cold_start_check",
    "watch_vault",
]
