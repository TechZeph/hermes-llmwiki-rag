"""User config precedence, init/doctor commands, starter vault, plugin fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from llmwiki.cli import main
from llmwiki.config import Settings
from llmwiki.setup import (
    create_starter_vault,
    discover_vaults,
    estimate_first_index,
    looks_like_vault,
)
from llmwiki.userconfig import config_path, load_user_config, save_user_config, update_user_config


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLMWIKI_CONFIG", str(tmp_path / "cfg" / "config.toml"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.delenv("LLMWIKI_VAULT", raising=False)
    monkeypatch.delenv("LLMWIKI_DB", raising=False)
    return tmp_path


def test_user_config_roundtrip_and_precedence(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert load_user_config() == {}
    assert not Settings.from_env().vault_configured
    vault = isolated / "v"
    vault.mkdir()
    path = save_user_config(
        {"vault": str(vault), "db": str(isolated / "db.sqlite"), "retrieval_mode": "lexical"}
    )
    assert path == config_path() and path.read_text().startswith("#")
    settings = Settings.from_env()
    assert settings.vault_configured and settings.vault_path == vault.resolve()
    assert (
        settings.db_path == (isolated / "db.sqlite").resolve()
        and settings.retrieval_mode == "lexical"
    )
    # Environment beats the file.
    other = isolated / "other"
    other.mkdir()
    monkeypatch.setenv("LLMWIKI_VAULT", str(other))
    assert Settings.from_env().vault_path == other.resolve()
    update_user_config({"default_profile": "history"})
    assert load_user_config()["default_profile"] == "history" and load_user_config()[
        "vault"
    ] == str(vault)
    (config_path()).write_text("not = [valid", encoding="utf-8")
    assert load_user_config() == {}


def test_starter_vault_and_discovery(tmp_path: Path) -> None:
    home = tmp_path / "home"
    target = home / "Documents" / "my-vault"
    created = create_starter_vault(target)
    assert looks_like_vault(created) and (created / "wiki" / "welcome.md").exists()
    assert "today" not in (created / "wiki" / "log.md").read_text()
    with pytest.raises(FileExistsError):
        create_starter_vault(target)
    (home / "Workspace" / "deep" / "vault2" / ".obsidian").mkdir(parents=True)
    found = discover_vaults(home)
    assert found[0].path == created  # most Markdown files first
    assert any(c.path.name == "vault2" for c in found)
    files, minutes = estimate_first_index(created)
    assert files >= 4 and minutes >= 1


def test_init_create_without_index_writes_config(isolated: Path) -> None:
    runner = CliRunner()
    target = isolated / "new-vault"
    result = runner.invoke(main, ["init", "--create", str(target), "--no-index", "--yes"])
    assert result.exit_code == 0, result.output
    assert "created starter vault" in result.output and "saved vault" in result.output
    assert load_user_config()["vault"] == str(target.resolve())
    # Non-interactive without a path is refused with guidance.
    result = runner.invoke(main, ["init", "--yes", "--no-index"], input="")
    assert result.exit_code != 0 and "pass a vault PATH or --create" in result.output
    # index now works without --vault
    show = runner.invoke(main, ["config", "show"])
    assert show.exit_code == 0 and str(target.resolve()) in show.output


def test_index_requires_configured_vault(isolated: Path) -> None:
    result = CliRunner().invoke(main, ["index", "--no-embed"])
    assert result.exit_code != 0 and "llmwiki init" in result.output


def test_doctor_reports_next_steps(isolated: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["doctor", "--json"])
    checks = {c["name"]: c for c in json.loads(result.stdout)}
    assert result.exit_code == 1
    assert checks["python"]["status"] == "ok" and checks["sqlite"]["status"] == "ok"
    assert checks["config"]["status"] == "fail" and "llmwiki init" in checks["config"]["next_step"]
    assert checks["projection"]["status"] == "skip" and checks["hermes"]["status"] == "skip"
    vault = create_starter_vault(isolated / "v")
    save_user_config({"vault": str(vault)})
    result = runner.invoke(main, ["doctor", "--json"])
    checks = {c["name"]: c for c in json.loads(result.stdout)}
    assert checks["config"]["status"] == "ok"
    assert (
        checks["projection"]["status"] == "fail"
        and "llmwiki index" in checks["projection"]["next_step"]
    )


def test_plugin_falls_back_to_user_config_and_slash_setup(isolated: Path) -> None:
    from hermes_plugin.tools import make_handlers
    from llmwiki.service import ServiceConfig, WikiService
    from tests.helpers import SAMPLE_KEYWORDS, KeywordEmbedder

    vault = create_starter_vault(isolated / "v")
    unconfigured = WikiService(
        ServiceConfig(vault=""),
        embedder_factory=lambda s: KeywordEmbedder(SAMPLE_KEYWORDS),
        gate_path=isolated / "none.json",
    )
    assert not unconfigured.configured and "llmwiki init" in unconfigured.status()["error"]
    save_user_config({"vault": str(vault)})
    fallback = WikiService(
        ServiceConfig(vault=""),
        embedder_factory=lambda s: KeywordEmbedder(SAMPLE_KEYWORDS),
        gate_path=isolated / "none.json",
    )
    assert fallback.configured and fallback.settings.vault_path == vault.resolve()
    saved: dict[str, str] = {}
    handlers = make_handlers(unconfigured, set_config=lambda k, v: saved.__setitem__(k, v))
    assert "not configured" in handlers.slash("status")
    assert handlers.slash("setup").startswith("usage:")
    assert "not an existing absolute directory" in handlers.slash("setup relative/path")
    out = handlers.slash(f"setup {vault}")
    assert out.startswith("vault set to v") and saved["vault"] == str(vault.resolve())
    assert "No projection yet" in out
    assert handlers.slash("doctor").startswith("[")
    assert "reindex completed" in handlers.slash("reindex")
    assert "documents" in handlers.slash("status")
    unconfigured.close()
    fallback.close()


def test_init_interactive_menu_creates_starter_vault(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(isolated / "home"))
    (isolated / "home").mkdir()
    runner = CliRunner()
    # No vaults found -> options: 1 enter a path, 2 create starter. Choose 2, decline indexing.
    result = runner.invoke(main, ["init", "--interactive"], input="2\nn\n")
    assert result.exit_code == 0, result.output
    assert "Looking for Obsidian vaults" in result.output
    assert "created starter vault" in result.output and "skipped indexing" in result.output
    created = (isolated / "home" / "llmwiki-vault").resolve()
    assert looks_like_vault(created) and load_user_config()["vault"] == str(created)
    # Now a vault exists under home -> it is listed first and chosen by default.
    result = runner.invoke(main, ["init", "--interactive", "--no-index"], input="\n")
    assert result.exit_code == 0, result.output
    assert "1. " in result.output and str(created) in result.output
