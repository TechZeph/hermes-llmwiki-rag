"""In-plugin vault watcher: keeps the projection fresh while the gateway runs.

Design constraints (see the vault plan, Stage 5):

- Never started from ``register(ctx)``: plugin doctor runs registration in
  a sandbox and a watcher there would be wrong. It starts lazily from the
  first hook invocation when ``watch: true`` is configured.
- Runs the observer and the incremental index passes on a daemon thread,
  never on a hook thread and never on the event loop. Runs are coalesced
  and never overlap (``llmwiki.watch.watch_vault``).
- Refuses a cold start: if the projection is missing or has chunks
  without vectors, the first full embed would hold gigabytes inside the
  gateway process, so the watcher declines and tells the operator to run
  ``llmwiki index`` once.
- Stops on plugin unload via ``ctx.on_unload`` and exposes its state to
  ``llmwiki_status``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llmwiki import db as dbmod
from llmwiki.config import Settings
from llmwiki.embeddings import Embedder
from llmwiki.models import IndexRunStats

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


class PluginWatcher:
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
        from llmwiki.watch import watch_vault

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


__all__ = ["COLD_START_MAX_MISSING", "PluginWatcher", "WatcherState", "cold_start_check"]
