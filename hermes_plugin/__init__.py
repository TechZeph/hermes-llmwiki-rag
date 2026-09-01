"""llmwiki — Hermes plugin exposing the local-first wiki RAG as explicit tools.

Directory-plugin contract: ``plugin.yaml`` next to this file and a
``register(ctx)`` entry point. The core retrieval package (``llmwiki``)
never imports Hermes; this package is the only adapter layer.

Install (development):

    ln -s /path/to/hermes-llmwiki-rag/hermes_plugin ~/.hermes/plugins/llmwiki
    ~/.hermes/hermes-agent/venv/bin/pip install -e /path/to/hermes-llmwiki-rag

Then set ``plugins.entries.llmwiki.settings.vault`` in ``~/.hermes/config.yaml``.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

__all__ = ["register"]

logger = logging.getLogger("llmwiki.plugin")

TOOLSET = "llmwiki"


def register(ctx: Any) -> None:
    """Register the llmwiki tools and the opt-in ``pre_llm_call`` hook."""
    from .runtime import PluginConfig, PluginRuntime
    from .schemas import REINDEX_SCHEMA, RELATED_SCHEMA, SEARCH_SCHEMA, STATUS_SCHEMA
    from .tools import make_handlers

    def get(key: str, default: Any) -> Any:
        try:
            return ctx.get_config(key, default)
        except Exception:  # config facade unavailable (e.g. doctor) -> defaults
            return default

    config = PluginConfig.from_getter(get)
    runtime = PluginRuntime(config)
    if not runtime.configured:
        logger.warning(
            "llmwiki plugin loaded without a usable vault setting; tools will report the problem"
        )
    handlers = make_handlers(runtime)

    ctx.register_tool(
        name="llmwiki_search",
        toolset=TOOLSET,
        schema=SEARCH_SCHEMA,
        handler=handlers.search,
        description=SEARCH_SCHEMA["description"],
        emoji="📚",
    )
    ctx.register_tool(
        name="llmwiki_status",
        toolset=TOOLSET,
        schema=STATUS_SCHEMA,
        handler=handlers.status,
        description=STATUS_SCHEMA["description"],
        emoji="🩺",
    )
    ctx.register_tool(
        name="llmwiki_reindex",
        toolset=TOOLSET,
        schema=REINDEX_SCHEMA,
        handler=handlers.reindex,
        description=REINDEX_SCHEMA["description"],
        emoji="🔄",
    )
    ctx.register_tool(
        name="llmwiki_related",
        toolset=TOOLSET,
        schema=RELATED_SCHEMA,
        handler=handlers.related,
        description=RELATED_SCHEMA["description"],
        emoji="🕸️",
    )
    # Always registered so the manifest and the runtime agree; it returns
    # None unless auto_inject is enabled and a calibrated gate exists.
    ctx.register_hook("pre_llm_call", handlers.pre_llm_call)
    ctx.register_hook("on_session_start", handlers.on_session_start)
    on_unload = getattr(ctx, "on_unload", None)
    if callable(on_unload):
        with contextlib.suppress(Exception):
            on_unload(runtime.close)
