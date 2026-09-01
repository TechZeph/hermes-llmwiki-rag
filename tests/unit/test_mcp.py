"""MCP server: tool registration and payload parity with the Hermes plugin."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from llmwiki.config import Settings
from llmwiki.indexer import Indexer
from llmwiki.mcp_server import build_server
from llmwiki.service import ServiceConfig
from tests.helpers import SAMPLE_KEYWORDS, SAMPLE_VAULT, KeywordEmbedder, write_vault

pytest.importorskip("mcp")


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("mcp")
    vault = write_vault(root / "vault", SAMPLE_VAULT)
    db = root / "proj" / "llmwiki.sqlite"
    Indexer(Settings(vault_path=vault, db_path=db), embedder=KeywordEmbedder(SAMPLE_KEYWORDS)).run()
    config = ServiceConfig(vault=str(vault), db=str(db), max_results=3)
    srv = build_server(
        config,
        embedder_factory=lambda s: KeywordEmbedder(SAMPLE_KEYWORDS),
        gate_path=root / "no-gate.json",
    )
    yield srv
    srv.llmwiki_service.close()


def _run(coro):
    return asyncio.run(coro)


def _text(result) -> str:
    # MCPServer.call_tool returns a CallToolResult with content blocks (SDK v2)
    # or a bare list of blocks (SDK v1).
    blocks = getattr(result, "content", None)
    if blocks is None:
        blocks = result[0] if isinstance(result, tuple) else result
    for block in blocks:
        text = getattr(block, "text", None)
        if text:
            return str(text)
    raise AssertionError(f"no text content in {result!r}")


def _schema(tool):
    return getattr(tool, "input_schema", None) or tool.inputSchema


def test_tools_are_registered(server) -> None:
    tools = _run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {"llmwiki_search", "llmwiki_status", "llmwiki_reindex"}
    search = next(t for t in tools if t.name == "llmwiki_search")
    assert "query" in _schema(search)["properties"]
    assert _schema(search).get("required") == ["query"]


def test_status_and_search_payloads(server) -> None:
    status = json.loads(_text(_run(server.call_tool("llmwiki_status", {}))))
    assert status["configured"] is True and status["integrity"]["ok"] is True
    assert status["counts"]["documents"] > 0
    search = json.loads(
        _text(
            _run(
                server.call_tool(
                    "llmwiki_search", {"query": "why did we choose sqlite-vec instead of faiss?"}
                )
            )
        )
    )
    assert search["intent"] == "decision"
    assert search["results"][0]["path"] == "wiki/projects/rag/decisions.md"
    assert search["context"].startswith("<<<UNTRUSTED RETRIEVED REFERENCE")
    assert len(search["results"]) <= 3
    assert not search["results"][0]["path"].startswith("/")
    empty = json.loads(_text(_run(server.call_tool("llmwiki_search", {"query": "  "}))))
    assert empty["error"]["type"] == "invalid-argument"
    bad = json.loads(
        _text(_run(server.call_tool("llmwiki_search", {"query": "x", "profile": "nope"})))
    )
    assert bad["error"]["type"] == "configuration"


def test_reindex_controls(server) -> None:
    full = json.loads(
        _text(_run(server.call_tool("llmwiki_reindex", {"mode": "full", "confirm": True})))
    )
    assert full["error"]["type"] == "not-permitted"
    inc = json.loads(_text(_run(server.call_tool("llmwiki_reindex", {"wait_seconds": 60}))))
    assert inc["state"] == "completed" and inc["job"]["result"]["errors"] == 0


def test_cli_mcp_command_exists(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from llmwiki.cli import main

    result = CliRunner().invoke(main, ["mcp", "--help"])
    assert result.exit_code == 0 and "stdio" in result.output
