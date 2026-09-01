"""Hermes-facing names for the host-agnostic :mod:`llmwiki.service`.

Kept as a thin alias module so the plugin's public surface is stable while
the engine lives in the core package (usable standalone, via MCP, or here).
"""

from __future__ import annotations

from llmwiki.service import (
    ConfigError,
    ReindexJob,
    ServiceConfig,
    WikiService,
    build_settings,
    validate_profile,
)

PluginConfig = ServiceConfig
PluginRuntime = WikiService

__all__ = [
    "ConfigError",
    "PluginConfig",
    "PluginRuntime",
    "ReindexJob",
    "ServiceConfig",
    "WikiService",
    "build_settings",
    "validate_profile",
]
