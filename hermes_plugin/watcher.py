"""Alias module: the in-host watcher lives in :mod:`llmwiki.watch`."""

from __future__ import annotations

from llmwiki.watch import COLD_START_MAX_MISSING, VaultWatcher, WatcherState, cold_start_check

PluginWatcher = VaultWatcher

__all__ = [
    "COLD_START_MAX_MISSING",
    "PluginWatcher",
    "VaultWatcher",
    "WatcherState",
    "cold_start_check",
]
