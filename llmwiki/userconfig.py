"""Persistent per-user configuration (``~/.config/llmwiki/config.toml``).

Precedence everywhere in the package is: explicit argument (CLI flag or
host setting) > ``LLMWIKI_*`` environment variable > this file > package
default. The file is written by ``llmwiki init`` and ``llmwiki config set``
and read by :meth:`llmwiki.config.Settings.from_env`, so the vault path
never has to be repeated on the command line and the Hermes plugin can
fall back to it when its own ``vault`` setting is unset.

The file is deliberately minimal TOML (flat string keys) so it can be
written without a TOML emitter dependency and read with ``tomllib``.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Final

KNOWN_KEYS: Final = (
    "vault",
    "db",
    "default_profile",
    "retrieval_mode",
    "embedding_model",
    "resource_profile",
    "embedding_batch_size",
    "embedding_memory_budget_mb",
    "embedding_min_available_mb",
    "embedding_threads",
)


def config_path() -> Path:
    """Return the user config path, honouring ``XDG_CONFIG_HOME`` and ``LLMWIKI_CONFIG``."""
    override = os.environ.get("LLMWIKI_CONFIG")
    if override:
        return Path(override).expanduser()
    from .sysinfo import user_config_dir

    return user_config_dir() / "config.toml"


def load_user_config(path: Path | None = None) -> dict[str, str]:
    """Return the flat string mapping from the config file, or ``{}``."""
    target = path or config_path()
    try:
        data = tomllib.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str | int | float) and str(v)}


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def save_user_config(values: Mapping[str, str], path: Path | None = None) -> Path:
    """Write the mapping as flat TOML with private permissions; returns the path."""
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(target.parent, 0o700)
    lines = ["# llmwiki user configuration (written by `llmwiki init` / `llmwiki config set`)"]
    for key in KNOWN_KEYS:
        if values.get(key):
            lines.append(f"{key} = {_toml_string(str(values[key]))}")
    for key, value in values.items():
        if key not in KNOWN_KEYS and value:
            lines.append(f"{key} = {_toml_string(str(value))}")
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if os.name == "posix":
        os.chmod(tmp, 0o600)
    os.replace(tmp, target)
    return target


def update_user_config(updates: Mapping[str, str], path: Path | None = None) -> Path:
    current = load_user_config(path)
    current.update({k: v for k, v in updates.items() if v is not None})
    return save_user_config(current, path)


def configured_vault(path: Path | None = None) -> Path | None:
    value = load_user_config(path).get("vault")
    return Path(value).expanduser() if value else None


__all__ = [
    "KNOWN_KEYS",
    "config_path",
    "configured_vault",
    "load_user_config",
    "save_user_config",
    "update_user_config",
]
