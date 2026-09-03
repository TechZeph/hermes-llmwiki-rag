"""Test the default user-data directory resolution."""

import os
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
    if os.name == "nt":
        assert _default_user_data_dir() == Path(os.environ["LOCALAPPDATA"]) / "llmwiki"
    else:
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


def test_from_env_reads_resource_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMWIKI_RESOURCE_PROFILE", "conservative")
    monkeypatch.setenv("LLMWIKI_EMBEDDING_BATCH_SIZE", "12")
    monkeypatch.setenv("LLMWIKI_EMBEDDING_MEMORY_BUDGET_MB", "768")
    monkeypatch.setenv("LLMWIKI_EMBEDDING_MIN_AVAILABLE_MB", "256")
    monkeypatch.setenv("LLMWIKI_EMBEDDING_THREADS", "2")

    settings = Settings.from_env()

    assert settings.resource_profile == "conservative"
    assert settings.embedding_batch_size == 12
    assert settings.embedding_memory_budget_mb == 768
    assert settings.embedding_min_available_mb == 256
    assert settings.embedding_threads == 2


def test_index_settings_require_a_vault_flag_or_environment_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Indexing must not silently treat the process working directory as a vault."""
    monkeypatch.delenv("LLMWIKI_VAULT", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(click.UsageError, match="pass --vault"):
        _resolve_settings(vault=None, db=None, watch=False, require_vault=True)
