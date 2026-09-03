"""Bundled Hermes skill registration and packaging contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import hermes_plugin


class SkillContext:
    def __init__(self) -> None:
        self.skills: list[tuple[str, Path, str | None]] = []

    def get_config(self, _key: str, default: Any = None) -> Any:
        return default

    def register_tool(self, **_kwargs: Any) -> None:
        return None

    def register_hook(self, _name: str, _callback: Any) -> None:
        return None

    def register_skill(self, name: str, path: Path, description: str | None = None) -> None:
        self.skills.append((name, Path(path), description))

    def on_unload(self, _callback: Any) -> None:
        return None


def test_plugin_registers_bundled_usage_skill() -> None:
    ctx = SkillContext()

    hermes_plugin.register(ctx)

    assert len(ctx.skills) == 1
    name, path, description = ctx.skills[0]
    assert name == "using-llmwiki"
    assert path.name == "SKILL.md"
    assert path.exists()
    assert description and "profiles" in description


def test_usage_skill_has_valid_frontmatter_and_safety_contract() -> None:
    path = Path(hermes_plugin.__file__).parent / "skills" / "using-llmwiki" / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _, frontmatter, body = text.split("---", 2)
    metadata = yaml.safe_load(frontmatter)

    assert metadata["name"] == "using-llmwiki"
    assert metadata["description"]
    assert "untrusted reference material" in body
    assert "`answer`" in body
    assert "`project:<id>`" in body
    assert "`history`" in body
    assert "`evidence`" in body
    assert "full rebuild" in body
