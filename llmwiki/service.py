"""Wiki service: the host-agnostic engine behind the CLI, MCP server, and Hermes plugin.

It validates host configuration into an immutable
:class:`llmwiki.config.Settings`, lazily loads the local models, opens one
SQLite connection per call, tracks the single background reindex job, and
owns the optional vault watcher. Every public method returns plain
dictionaries that a tool layer serialises; none of them raise for expected
operational conditions. Hosts (Hermes plugin, MCP server) are thin
adapters over :class:`WikiService`.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import db as dbmod
from .citations import build_context
from .confidence import Decision, InjectionGate, decide, load_gate
from .config import Settings
from .embeddings import Embedder
from .indexer import Indexer
from .models import RetrievalResult
from .reranker import Reranker
from .retrieval import VALID_MODES, Retriever, context_for
from .routing import route_query
from .watch import VaultWatcher

logger = logging.getLogger("llmwiki.service")

VALID_PROFILE_PREFIXES = ("answer", "evidence", "history", "all", "project:")


class ConfigError(ValueError):
    """Raised when plugin settings cannot produce a usable core configuration."""


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    vault: str
    db: str | None = None
    default_profile: str = "answer"
    retrieval_mode: str = "hybrid"
    max_results: int = 6
    context_budget_tokens: int = 1500
    rerank: bool = False
    allow_reindex: bool = True
    allow_full_rebuild: bool = False
    stale_after_hours: int = 24
    auto_inject: bool = False
    auto_inject_profile: str = "answer"
    auto_inject_deadline_ms: int = 1500
    auto_inject_budget_tokens: int = 800
    watch: bool = False
    watch_debounce_s: int = 2

    @staticmethod
    def from_getter(get: Callable[[str, Any], Any]) -> ServiceConfig:
        def _int(key: str, default: int, lo: int, hi: int) -> int:
            try:
                value = int(get(key, default))
            except (TypeError, ValueError):
                value = default
            return max(lo, min(hi, value))

        def _bool(key: str, default: bool) -> bool:
            value = get(key, default)
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)

        def _str(key: str, default: str) -> str:
            value = get(key, default)
            return default if value is None else str(value)

        return ServiceConfig(
            vault=_str("vault", ""),
            db=(get("db", None) or None),
            default_profile=_str("default_profile", "answer"),
            retrieval_mode=_str("retrieval_mode", "hybrid"),
            max_results=_int("max_results", 6, 1, 20),
            context_budget_tokens=_int("context_budget_tokens", 1500, 200, 8000),
            rerank=_bool("rerank", False),
            allow_reindex=_bool("allow_reindex", True),
            allow_full_rebuild=_bool("allow_full_rebuild", False),
            stale_after_hours=_int("stale_after_hours", 24, 1, 24 * 365),
            auto_inject=_bool("auto_inject", False),
            auto_inject_profile=_str("auto_inject_profile", "answer"),
            auto_inject_deadline_ms=_int("auto_inject_deadline_ms", 1500, 100, 2000),
            auto_inject_budget_tokens=_int("auto_inject_budget_tokens", 800, 100, 4000),
            watch=_bool("watch", False),
            watch_debounce_s=_int("watch_debounce_s", 2, 1, 600),
        )


def validate_profile(profile: str) -> str:
    profile = (profile or "").strip()
    if profile in ("answer", "evidence", "history", "all"):
        return profile
    if profile.startswith("project:") and len(profile) > len("project:"):
        return profile
    raise ConfigError("profile must be one of answer, evidence, history, all, or project:<id>")


def build_settings(config: ServiceConfig) -> Settings:
    vault_value = config.vault
    if not vault_value:
        from .userconfig import configured_vault

        fallback = configured_vault()
        if fallback is not None:
            vault_value = str(fallback)
    if not vault_value:
        raise ConfigError(
            "vault is not set: run `llmwiki init`, or set "
            "plugins.entries.llmwiki.settings.vault (Hermes) / --vault (CLI)"
        )
    vault = Path(vault_value).expanduser()
    if not vault.is_absolute():
        raise ConfigError("vault must be an absolute path")
    vault = vault.resolve()
    if not vault.is_dir():
        raise ConfigError("vault path is not a directory")
    if config.retrieval_mode not in VALID_MODES:
        raise ConfigError(f"retrieval_mode must be one of {VALID_MODES}")
    validate_profile(config.default_profile)
    validate_profile(config.auto_inject_profile)
    db_path = Path(config.db).expanduser().resolve() if config.db else Settings.from_env().db_path
    return replace(
        Settings(vault_path=vault, db_path=db_path),
        retrieval_mode=config.retrieval_mode,
        reranker_enabled=config.rerank,
        retrieval_top_k_final=config.max_results,
        context_budget_tokens=config.context_budget_tokens,
    )


@dataclass(slots=True)
class ReindexJob:
    mode: str
    started_at: float
    finished_at: float | None = None
    state: str = "running"
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": round((self.finished_at or time.time()) - self.started_at, 1),
            "result": dict(self.result),
            "error": self.error,
        }


def _safe_message(exc: BaseException, vault: Path | None) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if vault is not None:
        text = text.replace(str(vault), "<vault>")
    return text[:500]


class WikiService:
    """Long-lived engine shared by the tool handlers and the hook."""

    def __init__(
        self,
        config: ServiceConfig,
        *,
        embedder_factory: Callable[[Settings], Embedder] | None = None,
        reranker_factory: Callable[[Settings], Reranker] | None = None,
        gate_path: Path | None = None,
    ) -> None:
        self.config = config
        self._settings: Settings | None = None
        self._config_error: str = ""
        try:
            self._settings = build_settings(config)
        except ConfigError as exc:
            self._config_error = str(exc)
        self._embedder_factory = embedder_factory or _default_embedder
        self._reranker_factory = reranker_factory or _default_reranker
        self._embedder: Embedder | None = None
        self._reranker: Reranker | None = None
        self._model_lock = threading.Lock()
        self._job_lock = threading.Lock()
        self._job: ReindexJob | None = None
        self._job_thread: threading.Thread | None = None
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="llmwiki-inject"
        )
        self._decisions: deque[dict[str, Any]] = deque(maxlen=50)
        self._gate_path = gate_path or (Path(__file__).resolve().parent / "injection_gate.json")
        self._gate: InjectionGate | None = load_gate(self._gate_path)
        self._watcher: VaultWatcher | None = None
        self._watcher_lock = threading.Lock()
        self._schema_ready = False

    # --- lifecycle ------------------------------------------------------------

    def reconfigure(self, **changes: Any) -> dict[str, Any]:
        """Replace host settings (e.g. a new vault) and rebuild core settings."""
        self.config = replace(self.config, **changes)
        self._config_error = ""
        try:
            self._settings = build_settings(self.config)
        except ConfigError as exc:
            self._settings = None
            self._config_error = str(exc)
        self._schema_ready = False
        return {"configured": self.configured, "error": self._config_error}

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            raise ConfigError(self._config_error or "plugin is not configured")
        return self._settings

    @property
    def configured(self) -> bool:
        return self._settings is not None

    def close(self) -> None:
        with self._watcher_lock:
            watcher = self._watcher
        if watcher is not None:
            watcher.stop()
        self._pool.shutdown(wait=False, cancel_futures=True)

    # --- watcher ----------------------------------------------------------------

    def ensure_watcher(self) -> dict[str, Any] | None:
        """Start the in-plugin watcher once when configured; never raises.

        Called from hook callbacks (first turn), never from ``register``.
        """
        if not self.config.watch or not self.configured:
            return None
        with self._watcher_lock:
            if self._watcher is None:
                self._watcher = VaultWatcher(
                    self.settings,
                    embedder_factory=self._embedder_factory,
                    debounce_s=float(self.config.watch_debounce_s),
                )
            watcher = self._watcher
        try:
            watcher.start()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("llmwiki watcher failed to start: %s", type(exc).__name__)
        return watcher.state()

    def watcher_state(self) -> dict[str, Any] | None:
        with self._watcher_lock:
            watcher = self._watcher
        if watcher is None:
            return {"state": "disabled" if not self.config.watch else "not-started"}
        return watcher.state()

    def _get_embedder(self) -> Embedder:
        with self._model_lock:
            if self._embedder is None:
                self._embedder = self._embedder_factory(self.settings)
            return self._embedder

    def _get_reranker(self) -> Reranker | None:
        if not self.settings.reranker_enabled:
            return None
        with self._model_lock:
            if self._reranker is None:
                self._reranker = self._reranker_factory(self.settings)
            return self._reranker

    # --- retrieval --------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        profile: str | None = None,
        mode: str | None = None,
        top_k: int | None = None,
        rerank: bool | None = None,
    ) -> RetrievalResult:
        settings = self.settings
        profile = validate_profile(profile or self.config.default_profile)
        mode = mode or settings.retrieval_mode
        if mode not in VALID_MODES:
            raise ConfigError(f"mode must be one of {VALID_MODES}")
        embedder = self._get_embedder() if mode != "lexical" else None
        reranker = (
            self._get_reranker()
            if (rerank if rerank is not None else settings.reranker_enabled)
            else None
        )
        with dbmod.connect(settings.db_path) as conn:
            if not self._schema_ready:
                dbmod.init_schema(conn)
                self._schema_ready = True
            retriever = Retriever(conn, embedder=embedder, settings=settings, reranker=reranker)
            return retriever.retrieve(
                query,
                profile=profile,
                mode=mode,
                top_k=min(max(top_k or settings.retrieval_top_k_final, 1), 20),
                rerank=reranker is not None,
            )

    def search(
        self,
        query: str,
        *,
        profile: str | None = None,
        mode: str | None = None,
        max_results: int | None = None,
        include_context: bool = True,
    ) -> dict[str, Any]:
        result = self.retrieve(query, profile=profile, mode=mode, top_k=max_results)
        block = context_for(result, self.settings)
        payload: dict[str, Any] = {
            "query": result.query,
            "profile": result.profile,
            "mode": result.mode,
            "intent": result.intent,
            "conflicts": list(result.conflicts),
            "elapsed_ms": round(result.elapsed_ms, 1),
            "results": [
                {
                    "rank": i + 1,
                    "path": c.path,
                    "title": c.title,
                    "breadcrumb": " > ".join(c.heading_path) or c.title,
                    "section": c.section_name,
                    "chunk": c.position,
                    "chunk_hash": c.text_hash[:16],
                    "authority": c.authority_class,
                    "authority_match": bool(c.authority_match),
                    "source_kind": c.source_kind,
                    "page_role": c.page_role,
                    "project": c.project_id,
                    "channels": {
                        "dense_rank": c.dense_rank,
                        "lexical_rank": c.lexical_rank,
                        "rrf_score": c.rrf_score,
                        "rerank_score": c.rerank_score,
                    },
                    "excerpt": c.text[:600] + (" …" if len(c.text) > 600 else ""),
                }
                for i, c in enumerate(result.candidates)
            ],
            "citations": [c.to_dict() for c in block.citations],
            "untrusted_reference": True,
        }
        if include_context:
            payload["context"] = block.text
            payload["context_tokens"] = block.total_tokens
        return payload

    # --- related pages ----------------------------------------------------------

    def related(self, path: str, *, limit: int = 20) -> dict[str, Any]:
        from .entities import related_pages

        settings = self.settings
        rel = (path or "").strip().lstrip("/")
        if not rel or ".." in rel.split("/"):
            raise ConfigError("path must be a vault-relative Markdown path")
        with dbmod.connect(settings.db_path) as conn:
            if not self._schema_ready:
                dbmod.init_schema(conn)
                self._schema_ready = True
            exists = conn.execute("SELECT title FROM documents WHERE path = ?", (rel,)).fetchone()
            if exists is None:
                return {"path": rel, "found": False, "related": []}
            pages = related_pages(conn, rel, limit=max(1, min(limit, 50)))
        return {
            "path": rel,
            "title": str(exists[0]),
            "found": True,
            "related": [
                {"path": p.path, "title": p.title, "relation": p.relation, "weight": p.weight}
                for p in pages
            ],
        }

    # --- status ---------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        if not self.configured:
            return {
                "configured": False,
                "error": self._config_error,
                "remediation": "run `llmwiki init` on this machine, or: hermes config set plugins.entries.llmwiki.settings.vault /path/to/vault",
            }
        settings = self.settings
        report = dbmod.inspect_integrity(settings.db_path, vault_path=settings.vault_path)
        report.pop("path", None)
        missing = report.get("documents_missing_on_disk")
        report["documents_missing_on_disk"] = len(missing) if isinstance(missing, list) else 0
        counts: dict[str, int] = {}
        meta: dict[str, str] = {}
        last_run: dict[str, Any] | None = None
        if report.get("exists"):
            try:
                with dbmod.connect(settings.db_path) as conn:
                    for table in ("documents", "chunks", "chunk_embeddings", "chunks_fts"):
                        try:
                            counts[table] = int(
                                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                            )
                        except Exception:
                            counts[table] = -1
                    try:
                        meta = {
                            str(k): str(v)
                            for k, v in conn.execute(
                                "SELECT key, value FROM projection_meta"
                            ).fetchall()
                            if str(k).startswith(("recipe.", "embedding."))
                        }
                    except Exception:
                        meta = {}
                    row = conn.execute(
                        "SELECT started_at_ns, finished_at_ns, mode, documents_added, documents_updated, "
                        "documents_removed, errors_json FROM index_runs ORDER BY id DESC LIMIT 1"
                    ).fetchone()
                    if row:
                        finished_ns = row[1]
                        age_s = (time.time_ns() - int(finished_ns)) / 1e9 if finished_ns else None
                        last_run = {
                            "mode": row[2],
                            "finished": finished_ns is not None,
                            "age_seconds": round(age_s, 1) if age_s is not None else None,
                            "added": row[3],
                            "updated": row[4],
                            "removed": row[5],
                            "errors": len(json.loads(row[6] or "[]")),
                        }
            except Exception as exc:
                report["error"] = _safe_message(exc, settings.vault_path)
        stale = (
            last_run is None
            or last_run.get("age_seconds") is None
            or last_run["age_seconds"] > self.config.stale_after_hours * 3600
        )
        remediation: list[str] = []
        if not report.get("exists"):
            remediation.append("run llmwiki_reindex (incremental) to build the projection")
        elif not report.get("ok"):
            remediation.append(
                "run llmwiki_reindex mode=full with confirm=true to repair the projection"
            )
        elif stale:
            remediation.append("projection is stale; run llmwiki_reindex (incremental)")
        job = None
        with self._job_lock:
            if self._job is not None:
                job = self._job.to_dict()
        return {
            "configured": True,
            "vault": settings.vault_path.name,
            "profile_default": self.config.default_profile,
            "retrieval_mode": settings.retrieval_mode,
            "reranker_enabled": settings.reranker_enabled,
            "auto_inject": self.config.auto_inject,
            "auto_inject_gate": (
                "absent"
                if self._gate is None
                else (
                    "certified"
                    if self._gate.metrics.get("gate_a_passed")
                    else (
                        "certified-safe-low-coverage"
                        if self._gate.metrics.get("safety_passed")
                        else "uncertified"
                    )
                )
            ),
            "offline_after_provisioning": True,
            "integrity": report,
            "counts": counts,
            "projection_meta": meta,
            "last_index_run": last_run,
            "stale": stale,
            "reindex_job": job,
            "watcher": self.watcher_state(),
            "recent_injection_decisions": list(self._decisions)[-5:],
            "remediation": remediation,
        }

    # --- reindex ----------------------------------------------------------------

    def reindex(
        self, *, mode: str = "incremental", confirm: bool = False, wait_seconds: int = 30
    ) -> dict[str, Any]:
        settings = self.settings
        if mode not in ("incremental", "full"):
            raise ConfigError("mode must be 'incremental' or 'full'")
        if not self.config.allow_reindex:
            raise ConfigError(
                "reindexing is disabled by plugins.entries.llmwiki.settings.allow_reindex"
            )
        if mode == "full":
            if not self.config.allow_full_rebuild:
                raise ConfigError(
                    "full rebuild is disabled; set allow_full_rebuild: true to permit it"
                )
            if not confirm:
                raise ConfigError("full rebuild requires confirm=true")
        with self._job_lock:
            if self._job is not None and self._job.state == "running":
                return {"state": "already-running", "job": self._job.to_dict()}
            job = ReindexJob(mode=mode, started_at=time.time())
            self._job = job

            def run() -> None:
                try:
                    embedder = self._get_embedder()
                    stats = Indexer(settings, embedder=embedder).run(mode=mode)
                    job.result = {
                        "documents_seen": stats.documents_seen,
                        "added": stats.documents_added,
                        "updated": stats.documents_updated,
                        "removed": stats.documents_removed,
                        "skipped": stats.documents_skipped,
                        "embeddings_built": stats.embeddings_built,
                        "embeddings_rebuilt": stats.embeddings_rebuilt,
                        "errors": len(stats.errors),
                    }
                    job.state = "completed" if not stats.errors else "completed-with-errors"
                except Exception as exc:
                    job.state = "failed"
                    job.error = _safe_message(exc, settings.vault_path)
                    logger.warning("llmwiki reindex failed: %s", job.error)
                finally:
                    job.finished_at = time.time()

            thread = threading.Thread(target=run, name="llmwiki-reindex", daemon=True)
            self._job_thread = thread
            thread.start()
        thread.join(timeout=max(0, min(int(wait_seconds), 300)))
        return {"state": job.state, "job": job.to_dict()}

    # --- automatic injection ----------------------------------------------------

    def auto_inject(
        self, user_message: str, *, known_projects: Mapping[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Return ``{"context": ...}`` or ``None``; never raises, never blocks past the deadline."""
        started = time.perf_counter()
        record: dict[str, Any] = {"ts": time.time(), "injected": False, "reason": ""}
        try:
            if not self.config.auto_inject or not self.configured:
                record["reason"] = "disabled"
                return None
            if self._gate is None:
                record["reason"] = "no-calibrated-gate"
                return None
            if not self._gate.metrics.get("safety_passed", False):
                record["reason"] = "gate-not-certified"
                return None
            route = route_query(
                user_message,
                default_profile=self.config.auto_inject_profile,
                known_projects=tuple(known_projects or ()),
            )
            record["route"] = route.reason
            if not route.retrieve:
                record["reason"] = f"route:{route.reason}"
                return None
            deadline = self.config.auto_inject_deadline_ms / 1000.0
            future = self._pool.submit(
                self.retrieve, user_message, profile=route.profile, rerank=False
            )
            try:
                result = future.result(timeout=deadline)
            except concurrent.futures.TimeoutError:
                record["reason"] = "timeout"
                return None
            decision: Decision = decide(result, self._gate)
            record["score"] = round(decision.score, 3)
            record["profile"] = route.profile
            if not decision.inject:
                record["reason"] = decision.reason
                return None
            block = build_context(
                result.candidates,
                conflicts=result.conflicts,
                total_budget_tokens=self.config.auto_inject_budget_tokens,
                per_document_budget_tokens=max(self.config.auto_inject_budget_tokens // 2, 100),
                max_excerpts=4,
                retrieval_mode=result.mode,
            )
            if block.empty:
                record["reason"] = "empty-context"
                return None
            record["injected"] = True
            record["reason"] = "injected"
            record["excerpts"] = len(block.citations)
            return {"context": block.text}
        except Exception as exc:
            record["reason"] = "error:" + type(exc).__name__
            logger.warning("llmwiki auto-inject failed open: %s", type(exc).__name__)
            return None
        finally:
            record["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
            self._decisions.append(record)


def _default_embedder(settings: Settings) -> Embedder:
    from .embeddings import FastEmbedEmbedder

    return FastEmbedEmbedder(model_name=settings.embedding_model)


def _default_reranker(settings: Settings) -> Reranker:
    from .reranker import FastEmbedReranker

    return FastEmbedReranker(model_name=settings.reranker_model)


__all__ = [
    "ConfigError",
    "ReindexJob",
    "ServiceConfig",
    "WikiService",
    "build_settings",
    "validate_profile",
]
