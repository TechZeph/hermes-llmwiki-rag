"""llmwiki — Hermes plugin exposing the local-first wiki RAG as explicit tools.

Directory-plugin contract: ``plugin.yaml`` next to this file and a
``register(ctx)`` entry point. The core retrieval package (``llmwiki``)
never imports Hermes; this package is the only adapter layer.

Install (development):

    ln -s /path/to/llmwiki-rag/hermes_plugin ~/.hermes/plugins/llmwiki
    ~/.hermes/hermes-agent/venv/bin/pip install -e /path/to/llmwiki-rag

Then set ``plugins.entries.llmwiki.settings.vault`` in ``~/.hermes/config.yaml``.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

__all__ = ["register"]

logger = logging.getLogger("llmwiki.plugin")

TOOLSET = "llmwiki"
SKILL_NAME = "using-llmwiki"
SKILL_DESCRIPTION = (
    "Operational guidance for choosing llmwiki profiles, retrieval modes, related-page exploration, "
    "freshness checks, citations, and safe reindex behavior."
)


def register(ctx: Any) -> None:
    """Register the llmwiki tools, bundled skill, and opt-in ``pre_llm_call`` hook."""
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
    set_config = getattr(ctx, "set_config", None)
    handlers = make_handlers(runtime, set_config=set_config if callable(set_config) else None)

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

    register_skill = getattr(ctx, "register_skill", None)
    if callable(register_skill):
        skill_path = Path(__file__).parent / "skills" / SKILL_NAME / "SKILL.md"
        register_skill(SKILL_NAME, skill_path, SKILL_DESCRIPTION)

    # Always registered so the manifest and the runtime agree; it returns
    # None unless auto_inject is enabled and a calibrated gate exists.
    register_command = getattr(ctx, "register_command", None)
    if callable(register_command):
        with contextlib.suppress(Exception):
            register_command(
                name="llmwiki",
                handler=handlers.slash,
                description="Wiki RAG: /llmwiki [status|setup <vault>|reindex|doctor]",
                args_hint="[status|setup <vault>|reindex|doctor]",
            )
    ctx.register_hook("pre_llm_call", handlers.pre_llm_call)
    ctx.register_hook("on_session_start", handlers.on_session_start)
    on_unload = getattr(ctx, "on_unload", None)
    if callable(on_unload):
        with contextlib.suppress(Exception):
            on_unload(runtime.close)
