"""Configuration for the RAG core.

Configuration is explicit and immutable once loaded. The CLI and the
Hermes plugin both construct a :class:`Settings` from environment
variables and CLI flags, then pass it to every component. No component
ever reads ``os.environ`` directly; that keeps the system testable
and the configuration surface small.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"env var {name}={raw!r} is not a valid integer") from exc


def _default_user_data_dir() -> Path:
    """Return the per-user data directory for the RAG.

    Honours ``XDG_DATA_HOME`` when set (Linux freedesktop standard);
    otherwise falls back to ``~/.local/share/llmwiki``. The directory
    is created on first use, not here.
    """
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "llmwiki"
    return Path.home() / ".local" / "share" / "llmwiki"


@dataclass(frozen=True, slots=True)
class Settings:
    """All configuration the RAG core needs to start.

    Construct via :meth:`from_env` (the CLI does this) or directly
    (tests do this). Instances are frozen so they can be safely shared
    across threads.
    """

    # Vault source
    vault_path: Path
    """Absolute path to the Obsidian vault root."""

    # Database
    db_path: Path
    """SQLite database file. Parent directory is created on connect."""

    # Indexer behaviour
    file_watch: bool = False
    """If True, watch the vault for changes and re-index incrementally."""

    ignored_dirs: tuple[str, ...] = (
        ".obsidian",
        ".trash",
        ".git",
    )
    """Directory names (relative to the vault root) that the indexer skips entirely."""

    ignored_globs: tuple[str, ...] = (
        "**/.*",  # hidden files (config, state)
    )
    """Glob patterns (matched against paths relative to the vault) that the indexer skips."""

    # Logging
    log_level: str = "INFO"
    log_format: str = "text"  # "text" or "json"

    # Phase 3+ placeholders (kept here so all settings live in one place)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    """FastEmbed model name. Default matches the plan's recommendation."""
    embedding_dim: int = 384
    """Dimensionality of the embedding model. Must match the actual model."""
    reranker_model: str = "BAAI/bge-reranker-base"
    """Cross-encoder reranker. Default matches the plan's recommendation."""
    retrieval_mode: str = "hybrid"
    """Default channel mix: ``dense``, ``lexical``, or ``hybrid``. Experiment default."""
    retrieval_top_k_dense: int = 50
    retrieval_top_k_lexical: int = 50
    retrieval_top_k_final: int = 10
    rrf_k: int = 20
    """Reciprocal-rank-fusion constant. Selected on the dev split (docs/evaluation.md)."""
    rrf_dense_weight: float = 1.0
    rrf_lexical_weight: float = 1.0
    """Per-channel RRF weights; equal weights selected jointly on v1 and v2 dev splits."""
    query_recipe: str = "query-v2-bge-instruction"
    """Query embedding recipe id, selected on dev (see ``llmwiki.recipes.QUERY_RECIPES``)."""
    graph_channel_enabled: bool = False
    graph_channel_weight: float = 0.5
    graph_channel_seed_documents: int = 5
    graph_channel_max_neighbours: int = 30
    """V2 experiment: a third RRF channel of chunks from pages linked to/from the top fused pages."""
    multiquery: bool = False
    """V3 experiment: split multi-clause questions into sub-queries and fuse the results."""
    recency_boost: bool = False
    """V2 experiment: for current-state intent, order tier-0 pages by modification time."""
    project_graph_expansion: bool = True
    project_graph_hops: int = 1
    project_graph_max_linked: int = 40
    """``project:<id>`` admits curated pages linked from the workspace (V2, measured)."""
    context_budget_tokens: int = 1500
    context_per_document_tokens: int = 600
    context_max_excerpts: int = 8
    """Context builder budgets (estimated tokens) for tool responses and injection."""
    max_chunks_per_document: int = 3
    """Document diversification cap applied after fusion (0 disables)."""
    reranker_enabled: bool = False
    """Cross-encoder reranking ships only when the held-out gate passes."""
    rerank_candidates: int = 30
    """How many fused candidates the reranker scores when enabled."""

    @staticmethod
    def from_env() -> Settings:
        """Build settings from ``LLMWIKI_*`` environment variables.

        Recognised variables:

        - ``LLMWIKI_VAULT`` (default: current working directory)
        - ``LLMWIKI_DB``   (default: ``$XDG_DATA_HOME/llmwiki/llmwiki.sqlite``
          or ``~/.local/share/llmwiki/llmwiki.sqlite``)
        - ``LLMWIKI_FILE_WATCH`` (default: "0")
        - ``LLMWIKI_LOG_LEVEL``  (default: "INFO")
        - ``LLMWIKI_LOG_FORMAT`` (default: "text")
        """
        vault = Path(_env("LLMWIKI_VAULT", os.getcwd())).expanduser().resolve()
        default_db = _default_user_data_dir() / "llmwiki.sqlite"
        db = Path(_env("LLMWIKI_DB", str(default_db))).expanduser().resolve()
        watch_raw = _env("LLMWIKI_FILE_WATCH", "0").lower()
        watch = watch_raw in ("1", "true", "yes", "on")
        return Settings(
            vault_path=vault,
            db_path=db,
            file_watch=watch,
            log_level=_env("LLMWIKI_LOG_LEVEL", "INFO"),
            log_format=_env("LLMWIKI_LOG_FORMAT", "text"),
        )


__all__ = ["Settings"]
