"""Hermes plugin stub (Phase 8 placeholder).

The plugin will register ``llmwiki_search``, ``llmwiki_status``, and
``llmwiki_reindex`` as Hermes tools, and connect the RAG core's
``pre_llm_call`` hook to automatic context injection.

This module exists now so the package shape matches the plan. The
real implementation lands in Phase 8 once the Hermes plugin API is
confirmed.
"""

from __future__ import annotations

__all__: list[str] = []
