"""Tool handlers and hook callbacks: thin, defensive adapters over the runtime.

Handlers follow the Hermes convention ``handler(args: dict, **kw) -> str``
and always return a JSON document. Expected operational problems become
``{"error": {...}}`` payloads rather than exceptions, so the model gets an
actionable message and the agent loop is never interrupted.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .runtime import ConfigError, PluginRuntime


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _error(kind: str, message: str, **extra: Any) -> str:
    return _dump({"error": {"type": kind, "message": message, **extra}})


def _coerce_int(value: Any, default: int | None, lo: int, hi: int) -> int | None:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, number))


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class Handlers:
    def __init__(
        self, runtime: PluginRuntime, *, set_config: Callable[[str, Any], None] | None = None
    ) -> None:
        self.runtime = runtime
        self._set_config = set_config

    def search(self, args: dict[str, Any], **_: Any) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return _error("invalid-argument", "query is required")
        try:
            payload = self.runtime.search(
                query,
                profile=(str(args["profile"]).strip() if args.get("profile") else None),
                mode=(str(args["mode"]).strip() if args.get("mode") else None),
                max_results=_coerce_int(args.get("max_results"), None, 1, 20),
                include_context=_coerce_bool(args.get("include_context"), True),
            )
        except ConfigError as exc:
            return _error("configuration", str(exc))
        except Exception as exc:  # never leak paths or tracebacks
            return _error("retrieval-failed", f"{type(exc).__name__}", hint="run llmwiki_status")
        return _dump(payload)

    def status(self, args: dict[str, Any], **_: Any) -> str:
        try:
            return _dump(self.runtime.status())
        except Exception as exc:
            return _error("status-failed", type(exc).__name__)

    def reindex(self, args: dict[str, Any], **_: Any) -> str:
        mode = str(args.get("mode") or "incremental").strip().lower()
        try:
            payload = self.runtime.reindex(
                mode=mode,
                confirm=_coerce_bool(args.get("confirm"), False),
                wait_seconds=_coerce_int(args.get("wait_seconds"), 30, 0, 300) or 0,
            )
        except ConfigError as exc:
            return _error(
                "not-permitted"
                if "disabled" in str(exc) or "confirm" in str(exc)
                else "configuration",
                str(exc),
            )
        except Exception as exc:
            return _error("reindex-failed", type(exc).__name__)
        return _dump(payload)

    def related(self, args: dict[str, Any], **_: Any) -> str:
        path = str(args.get("path") or "").strip()
        if not path:
            return _error("invalid-argument", "path is required")
        try:
            payload = self.runtime.related(
                path, limit=_coerce_int(args.get("limit"), 20, 1, 50) or 20
            )
        except ConfigError as exc:
            return _error("configuration", str(exc))
        except Exception as exc:
            return _error("related-failed", type(exc).__name__)
        return _dump(payload)

    def slash(self, raw_args: str = "", **_: Any) -> str:
        """``/llmwiki [status|setup <vault>|reindex|doctor]`` for CLI and gateway sessions."""
        parts = (raw_args or "").strip().split(maxsplit=1)
        verb = parts[0].lower() if parts else "status"
        arg = parts[1].strip() if len(parts) > 1 else ""
        if verb == "setup":
            if not arg:
                return "usage: /llmwiki setup /absolute/path/to/vault"
            from pathlib import Path

            vault = Path(arg).expanduser()
            if not vault.is_absolute() or not vault.is_dir():
                return f"not an existing absolute directory: {arg}"
            outcome = self.runtime.reconfigure(vault=str(vault.resolve()))
            if not outcome["configured"]:
                return f"could not use that vault: {outcome['error']}"
            persisted = ""
            if self._set_config is not None:
                try:
                    self._set_config("vault", str(vault.resolve()))
                    persisted = " and saved to plugins.entries.llmwiki.settings.vault"
                except Exception as exc:
                    persisted = f" (not saved to Hermes config: {type(exc).__name__})"
            status = self.runtime.status()
            hint = (
                ""
                if status.get("integrity", {}).get("exists")
                else " No projection yet: run `llmwiki index` or `/llmwiki reindex`."
            )
            return f"vault set to {vault.resolve().name}{persisted}.{hint}"
        if verb == "reindex":
            try:
                payload = self.runtime.reindex(mode="incremental", confirm=False, wait_seconds=120)
            except ConfigError as exc:
                return f"cannot reindex: {exc}"
            job = payload.get("job", {})
            return f"reindex {payload.get('state')}: {job.get('result') or job.get('error') or ''}"
        if verb == "doctor":
            from llmwiki.doctor import format_checks, run_doctor

            return format_checks(run_doctor())
        status = self.runtime.status()
        if not status.get("configured"):
            return f"llmwiki: not configured ({status.get('error')}). Try /llmwiki setup <vault>."
        counts = status.get("counts", {})
        integrity = status.get("integrity", {})
        watcher = status.get("watcher", {})
        return (
            f"llmwiki: vault {status.get('vault')}, {counts.get('documents', 0)} documents, "
            f"{counts.get('chunks', 0)} chunks, integrity {'ok' if integrity.get('ok') else 'PROBLEM'}, "
            f"{'stale' if status.get('stale') else 'fresh'}, watcher {watcher.get('state')}, "
            f"auto-inject {'on' if status.get('auto_inject') else 'off'} "
            f"({status.get('auto_inject_gate')})."
        )

    def on_session_start(self, **kwargs: Any) -> None:
        """Start optional background services on first session; never returns content."""
        self.runtime.ensure_watcher()
        self.runtime.start_update_check()
        return None

    def pre_llm_call(self, user_message: str = "", **kwargs: Any) -> dict[str, str] | None:
        """Opt-in automatic injection. Uses only the current user message."""
        self.runtime.ensure_watcher()
        if not isinstance(user_message, str) or not user_message:
            return None
        return self.runtime.auto_inject(user_message)


def make_handlers(
    runtime: PluginRuntime, *, set_config: Callable[[str, Any], None] | None = None
) -> Handlers:
    return Handlers(runtime, set_config=set_config)


HandlerFn = Callable[..., str]

__all__ = ["Handlers", "make_handlers"]
