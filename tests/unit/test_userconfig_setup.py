"""User config precedence, init/doctor commands, starter vault, plugin fallback."""

from __future__ import annotations

import json
import os
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
    import shutil

    home = isolated / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(shutil, "which", lambda name: None)  # no Obsidian prompt
    home.mkdir()
    runner = CliRunner()
    # No vaults found -> options: 1 enter a path, 2 create starter. Choose 2, decline indexing.
    result = runner.invoke(main, ["init", "--interactive"], input="2\nn\n")
    assert result.exit_code == 0, result.output
    assert "Looking for Obsidian vaults" in result.output
    assert "created starter vault" in result.output and "skipped indexing" in result.output
    created = (home / "llmwiki-vault").resolve()
    assert looks_like_vault(created) and load_user_config()["vault"] == str(created)
    # Now a vault exists under home -> it is listed first and chosen by default.
    result = runner.invoke(main, ["init", "--interactive", "--no-index"], input="\n")
    assert result.exit_code == 0, result.output
    assert "1. " in result.output and str(created) in result.output


def test_obsidian_detection_and_open_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from llmwiki.setup import find_obsidian, obsidian_open_command, open_in_obsidian

    monkeypatch.setattr(shutil, "which", lambda name: None)
    import sys

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert find_obsidian() is None
    assert open_in_obsidian(tmp_path) is False
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/obsidian" if name == "obsidian" else None
    )
    assert find_obsidian() == "/usr/bin/obsidian"
    cmd = obsidian_open_command("/usr/bin/obsidian", tmp_path / "my vault")
    assert (
        cmd[0] == "/usr/bin/obsidian"
        and cmd[1].startswith("obsidian://open?path=")
        and "%20vault" in cmd[1]
    )
    mac = obsidian_open_command("open -a Obsidian", tmp_path)
    assert mac[:3] == ["open", "-a", "Obsidian"]


def test_init_starter_mentions_obsidian(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    result = CliRunner().invoke(
        main, ["init", "--create", str(isolated / "nv"), "--no-index", "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert "Obsidian is optional" in result.output and "obsidian.md/download" in result.output


@pytest.mark.skipif(
    os.name == "nt", reason="install.sh is POSIX-only; install.ps1 is tested in Windows CI"
)
def test_install_script_dry_run(tmp_path: Path) -> None:
    import os
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    env = dict(
        os.environ,
        HOME=str(tmp_path),
        LLMWIKI_INSTALL_DIR=str(tmp_path / "venv"),
        LLMWIKI_BIN_DIR=str(tmp_path / "bin"),
        HERMES_HOME=str(tmp_path / "no-hermes"),
    )
    out = subprocess.run(
        ["bash", str(repo / "install.sh"), "--dry-run", "--no-init"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout
    assert "package source: checkout" in out and "would run:" in out and "pip install" in out
    assert "skipped init" in out
    syntax = subprocess.run(
        ["bash", "-n", str(repo / "install.sh")], capture_output=True, text=True
    )
    assert syntax.returncode == 0, syntax.stderr
    hermes = subprocess.run(
        ["bash", str(repo / "install.sh"), "--dry-run", "--no-init", "--hermes"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert hermes.returncode == 1 and "Hermes venv not found" in hermes.stderr


def test_windows_style_paths_and_non_posix_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llmwiki import db as dbmod
    from llmwiki.sysinfo import peak_rss_mb, user_config_dir, user_data_dir

    assert peak_rss_mb() >= 0.0
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    import llmwiki.sysinfo as sysinfo

    monkeypatch.setattr(sysinfo, "_IS_POSIX", False)
    monkeypatch.setattr(sysinfo, "_IS_WINDOWS", True)
    monkeypatch.setattr(dbmod, "_POSIX", False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    assert user_config_dir() == tmp_path / "AppData" / "Roaming" / "llmwiki"
    assert user_data_dir() == tmp_path / "AppData" / "Local" / "llmwiki"
    # Non-POSIX path still creates the projection directory before connecting.
    db_path = tmp_path / "AppData" / "Local" / "llmwiki" / "llmwiki.sqlite"
    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    assert db_path.exists()
    assert peak_rss_mb() >= 0.0  # nt branch without psapi on Linux returns 0.0
