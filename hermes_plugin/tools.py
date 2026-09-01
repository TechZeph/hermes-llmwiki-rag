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
    def __init__(self, runtime: PluginRuntime) -> None:
        self.runtime = runtime

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

    def pre_llm_call(self, user_message: str = "", **kwargs: Any) -> dict[str, str] | None:
        """Opt-in automatic injection. Uses only the current user message."""
        if not isinstance(user_message, str) or not user_message:
            return None
        return self.runtime.auto_inject(user_message)


def make_handlers(runtime: PluginRuntime) -> Handlers:
    return Handlers(runtime)


HandlerFn = Callable[..., str]

__all__ = ["Handlers", "make_handlers"]
