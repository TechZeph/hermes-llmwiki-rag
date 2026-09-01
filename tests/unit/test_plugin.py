"""Hermes plugin contract: registration, tool payloads, permissions, hook safety."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import hermes_plugin
from hermes_plugin.runtime import ConfigError, PluginConfig, PluginRuntime, build_settings
from llmwiki.confidence import InjectionGate
from llmwiki.config import Settings
from llmwiki.indexer import Indexer
from tests.helpers import SAMPLE_KEYWORDS, SAMPLE_VAULT, KeywordEmbedder, write_vault


class FakeContext:
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self.tools: dict[str, dict[str, Any]] = {}
        self.hooks: dict[str, list[Any]] = {}
        self.unload: list[Any] = []

    def get_config(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def register_tool(
        self, *, name: str, toolset: str, schema: dict[str, Any], handler: Any, **kw: Any
    ) -> None:
        assert schema["name"] == name
        assert "parameters" in schema
        self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler, **kw}

    def register_hook(self, hook_name: str, callback: Any) -> None:
        self.hooks.setdefault(hook_name, []).append(callback)

    def on_unload(self, callback: Any) -> None:
        self.unload.append(callback)


@pytest.fixture(scope="module")
def vault_and_db(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("plugin")
    vault = write_vault(root / "vault", SAMPLE_VAULT)
    db = root / "proj" / "llmwiki.sqlite"
    settings = Settings(vault_path=vault, db_path=db)
    stats = Indexer(settings, embedder=KeywordEmbedder(SAMPLE_KEYWORDS)).run()
    assert not stats.errors
    return vault, db


def _runtime(vault: Path, db: Path, **overrides: Any) -> PluginRuntime:
    config = PluginConfig(vault=str(vault), db=str(db), **overrides)
    # Point at a non-existent gate so tests do not depend on the shipped calibration.
    return PluginRuntime(
        config,
        embedder_factory=lambda s: KeywordEmbedder(SAMPLE_KEYWORDS),
        gate_path=db.parent / "no-gate.json",
    )


def test_register_declares_manifest_tools_and_hook(vault_and_db) -> None:
    vault, db = vault_and_db
    ctx = FakeContext({"vault": str(vault), "db": str(db)})
    hermes_plugin.register(ctx)
    manifest = (Path(hermes_plugin.__file__).parent / "plugin.yaml").read_text()
    assert set(ctx.tools) == {"llmwiki_search", "llmwiki_status", "llmwiki_reindex"}
    for name in ctx.tools:
        assert f"- {name}" in manifest
        assert ctx.tools[name]["toolset"] == "llmwiki"
        assert "override" not in ctx.tools[name] or not ctx.tools[name]["override"]
    assert list(ctx.hooks) == ["pre_llm_call"]
    assert "- pre_llm_call" in manifest
    hook = ctx.hooks["pre_llm_call"][0]
    # Doctor requires **kwargs for forward compatibility.
    import inspect

    assert any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(hook).parameters.values()
    )
    assert ctx.unload


def test_register_without_vault_still_loads_and_reports(tmp_path: Path) -> None:
    ctx = FakeContext({})
    hermes_plugin.register(ctx)
    status = json.loads(ctx.tools["llmwiki_status"]["handler"]({}))
    assert status["configured"] is False
    assert "vault" in status["error"]
    search = json.loads(ctx.tools["llmwiki_search"]["handler"]({"query": "x"}))
    assert search["error"]["type"] == "configuration"


def test_build_settings_validates(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        build_settings(PluginConfig(vault="relative/path"))
    with pytest.raises(ConfigError):
        build_settings(PluginConfig(vault=str(tmp_path / "missing")))
    with pytest.raises(ConfigError):
        build_settings(PluginConfig(vault=str(tmp_path), retrieval_mode="magic"))
    with pytest.raises(ConfigError):
        build_settings(PluginConfig(vault=str(tmp_path), default_profile="project:"))
    settings = build_settings(
        PluginConfig(vault=str(tmp_path), db=str(tmp_path / "x.sqlite"), max_results=4)
    )
    assert settings.retrieval_top_k_final == 4 and settings.retrieval_mode == "hybrid"


def test_search_tool_returns_bounded_cited_payload(vault_and_db) -> None:
    vault, db = vault_and_db
    handlers = hermes_plugin.tools.make_handlers(_runtime(vault, db, max_results=3))
    payload = json.loads(handlers.search({"query": "what is the current status of sqlite-vec?"}))
    assert payload["profile"] == "answer" and payload["mode"] == "hybrid"
    assert payload["intent"] == "current-state"
    assert 1 <= len(payload["results"]) <= 3
    top = payload["results"][0]
    assert top["path"] == "wiki/projects/rag/current-state.md"
    assert top["authority"] == "current-state" and top["authority_match"] is True
    assert not top["path"].startswith("/") and str(vault) not in json.dumps(payload)
    assert payload["untrusted_reference"] is True
    assert payload["context"].startswith("<<<UNTRUSTED RETRIEVED REFERENCE")
    assert payload["citations"][0]["path"] == top["path"]
    # explicit profile + lexical mode + no context
    ev = json.loads(
        handlers.search(
            {
                "query": "Condorcet",
                "profile": "evidence",
                "mode": "lexical",
                "include_context": False,
            }
        )
    )
    assert ev["results"][0]["path"] == "raw/papers/rrf-paper.md" and "context" not in ev
    bad = json.loads(handlers.search({"query": "x", "profile": "nope"}))
    assert bad["error"]["type"] == "configuration"
    assert json.loads(handlers.search({}))["error"]["type"] == "invalid-argument"


def test_status_tool_reports_projection_without_paths(vault_and_db) -> None:
    vault, db = vault_and_db
    handlers = hermes_plugin.tools.make_handlers(_runtime(vault, db))
    status = json.loads(handlers.status({}))
    assert status["configured"] is True
    assert status["vault"] == vault.name
    assert status["integrity"]["ok"] is True and status["integrity"]["schema_version"] == 7
    assert (
        status["counts"]["documents"] > 0
        and status["counts"]["chunks_fts"] == status["counts"]["chunks"]
    )
    assert status["last_index_run"]["finished"] is True and status["stale"] is False
    assert status["auto_inject"] is False and status["auto_inject_gate"] == "absent"
    assert "recipe.document_embedding" in status["projection_meta"]
    assert str(db) not in json.dumps(status) and str(vault) not in json.dumps(status)


def test_reindex_permissions_and_incremental_run(vault_and_db) -> None:
    vault, db = vault_and_db
    handlers = hermes_plugin.tools.make_handlers(_runtime(vault, db))
    full = json.loads(handlers.reindex({"mode": "full", "confirm": True}))
    assert full["error"]["type"] == "not-permitted"
    denied = hermes_plugin.tools.make_handlers(_runtime(vault, db, allow_reindex=False))
    assert json.loads(denied.reindex({}))["error"]["type"] == "not-permitted"
    allowed = hermes_plugin.tools.make_handlers(_runtime(vault, db, allow_full_rebuild=True))
    unconfirmed = json.loads(allowed.reindex({"mode": "full"}))
    assert (
        unconfirmed["error"]["type"] == "not-permitted"
        and "confirm" in unconfirmed["error"]["message"]
    )
    inc = json.loads(handlers.reindex({"wait_seconds": 60}))
    assert inc["state"] == "completed"
    assert inc["job"]["result"]["documents_seen"] > 0 and inc["job"]["result"]["errors"] == 0
    assert json.loads(handlers.reindex({"mode": "sideways"}))["error"]["type"] == "configuration"


def test_pre_llm_call_is_off_by_default_and_fails_open(vault_and_db) -> None:
    vault, db = vault_and_db
    handlers = hermes_plugin.tools.make_handlers(_runtime(vault, db))
    assert (
        handlers.pre_llm_call(
            user_message="what is the current status of sqlite-vec?", session_id="s"
        )
        is None
    )
    assert handlers.pre_llm_call(user_message="", conversation_history=[{"role": "user"}]) is None
    # Enabled but without a calibrated gate: still nothing.
    on = _runtime(vault, db, auto_inject=True)
    assert on.auto_inject("what is the current status of sqlite-vec?") is None
    assert on.status()["recent_injection_decisions"][-1]["reason"] == "no-calibrated-gate"


def test_pre_llm_call_injects_only_with_gate_and_route(vault_and_db, tmp_path: Path) -> None:
    vault, db = vault_and_db
    gate = InjectionGate(
        weights={"authority_top1": 4.0, "n_candidates_norm": 2.0},
        bias=-1.0,
        threshold=0.5,
        metrics={"gate_a_passed": True, "safety_passed": True},
    )
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate.to_dict()))
    config = PluginConfig(
        vault=str(vault), db=str(db), auto_inject=True, auto_inject_budget_tokens=400
    )
    runtime = PluginRuntime(
        config, embedder_factory=lambda s: KeywordEmbedder(SAMPLE_KEYWORDS), gate_path=gate_path
    )
    injected = runtime.auto_inject("what is the current status of sqlite-vec?")
    assert injected is not None and injected["context"].startswith(
        "<<<UNTRUSTED RETRIEVED REFERENCE"
    )
    assert "wiki/projects/rag/current-state.md" in injected["context"]
    assert runtime.status()["recent_injection_decisions"][-1]["injected"] is True
    assert runtime.auto_inject("thanks!") is None  # routed away
    uncertified = InjectionGate(weights=gate.weights, bias=gate.bias, threshold=0.5)
    (tmp_path / "weak.json").write_text(json.dumps(uncertified.to_dict()))
    weak = PluginRuntime(
        config,
        embedder_factory=lambda s: KeywordEmbedder(SAMPLE_KEYWORDS),
        gate_path=tmp_path / "weak.json",
    )
    assert weak.auto_inject("what is the current status of sqlite-vec?") is None
    assert weak.status()["recent_injection_decisions"][-1]["reason"] == "gate-not-certified"
    assert weak.status()["auto_inject_gate"] == "uncertified"
    assert runtime.status()["recent_injection_decisions"][-1]["reason"].startswith("route:")
    # Deadline: a zero-budget retrieval that cannot finish in time returns None.
    slow = PluginConfig(vault=str(vault), db=str(db), auto_inject=True, auto_inject_deadline_ms=100)

    class SlowEmbedder(KeywordEmbedder):
        def embed(self, texts):
            import time

            time.sleep(0.5)
            return super().embed(texts)

    slow_rt = PluginRuntime(
        slow, embedder_factory=lambda s: SlowEmbedder(SAMPLE_KEYWORDS), gate_path=gate_path
    )
    assert slow_rt.auto_inject("what is the current status of sqlite-vec?") is None
    assert slow_rt.status()["recent_injection_decisions"][-1]["reason"] == "timeout"
    # No query text is ever recorded in decisions.
    for record in slow_rt.status()["recent_injection_decisions"]:
        assert "sqlite-vec" not in json.dumps(record)
