"""Test-wide isolation: never read the developer's real user config or Hermes home."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_user_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMWIKI_CONFIG", str(tmp_path / "isolated-config.toml"))
    monkeypatch.delenv("LLMWIKI_VAULT", raising=False)
    monkeypatch.delenv("LLMWIKI_DB", raising=False)
