"""Test the default user-data directory resolution."""

from pathlib import Path

import click
import pytest

from llmwiki.cli import _resolve_settings
from llmwiki.config import Settings, _default_user_data_dir


def test_default_user_data_dir_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert _default_user_data_dir() == tmp_path / "llmwiki"


def test_default_user_data_dir_no_xdg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert _default_user_data_dir() == Path.home() / ".local" / "share" / "llmwiki"


def test_from_env_uses_user_data_dir_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.delenv("LLMWIKI_DB", raising=False)
    s = Settings.from_env()
    assert s.db_path == (tmp_path / "llmwiki" / "llmwiki.sqlite").resolve()


def test_from_env_honours_llmwiki_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMWIKI_DB", "/tmp/custom/llmwiki.sqlite")
    s = Settings.from_env()
    assert s.db_path == Path("/tmp/custom/llmwiki.sqlite").resolve()


def test_index_settings_require_a_vault_flag_or_environment_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Indexing must not silently treat the process working directory as a vault."""
    monkeypatch.delenv("LLMWIKI_VAULT", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(click.UsageError, match="pass --vault"):
        _resolve_settings(vault=None, db=None, watch=False, require_vault=True)
