"""``llmwiki doctor``: environment, configuration, projection, and Hermes checks.

Each check returns a :class:`Check` with a status (``ok``, ``warn``,
``fail``, ``skip``), a one-line detail, and the next command to run when
something is wrong. The CLI prints them and exits non-zero on any ``fail``.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import db as dbmod
from .config import Settings
from .userconfig import config_path, load_user_config


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: str  # ok | warn | fail | skip
    detail: str
    next_step: str = ""


def _python() -> Check:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 11)
    return Check(
        "python",
        "ok" if ok else "fail",
        f"{v.major}.{v.minor}.{v.micro}",
        "" if ok else "install Python 3.11 or newer",
    )


def _sqlite() -> Check:
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("SELECT vec_version()")
        return Check(
            "sqlite",
            "ok",
            f"{sqlite3.sqlite_version} with FTS5 and sqlite-vec {sqlite_vec.__version__}",
        )
    except Exception as exc:
        return Check(
            "sqlite",
            "fail",
            f"{type(exc).__name__}: {exc}",
            "use a Python build whose sqlite3 has FTS5 and `pip install sqlite-vec`",
        )


def _model(settings: Settings | None) -> Check:
    model = (
        settings.embedding_model
        if settings
        else Settings(vault_path=Path("/"), db_path=Path("/")).embedding_model
    )
    try:
        import fastembed  # noqa: F401
    except Exception:
        return Check("model", "fail", "fastembed not importable", "pip install fastembed")
    cache = Path(os.environ.get("FASTEMBED_CACHE_PATH") or Path.home() / ".cache" / "fastembed")
    slug = model.replace("/", "--").lower()
    present = cache.is_dir() and any(
        slug.split("--")[-1] in p.name.lower() for p in cache.iterdir()
    )
    if present:
        return Check("model", "ok", f"{model} cached in {cache}")
    return Check(
        "model",
        "warn",
        f"{model} not in {cache}; first `llmwiki index` downloads it (~130 MB) once",
        "run `llmwiki index` while online, or copy a provisioned cache (docs/operations.md)",
    )


def _config() -> tuple[Check, Settings | None]:
    path = config_path()
    values = load_user_config(path)
    env_vault = os.environ.get("LLMWIKI_VAULT")
    vault = env_vault or values.get("vault")
    if not vault:
        return (
            Check(
                "config",
                "fail",
                f"no vault configured ({path} absent, LLMWIKI_VAULT unset)",
                "run `llmwiki init`",
            ),
            None,
        )
    vault_path = Path(vault).expanduser()
    if not vault_path.is_dir():
        return (
            Check(
                "config",
                "fail",
                f"vault path does not exist: {vault_path}",
                "run `llmwiki init` to pick a vault",
            ),
            None,
        )
    settings = Settings.from_env()
    if settings.vault_path != vault_path.resolve():
        settings = Settings(vault_path=vault_path.resolve(), db_path=settings.db_path)
    source = "LLMWIKI_VAULT" if env_vault else str(path)
    note = (
        ""
        if (vault_path / ".obsidian").is_dir()
        else " (no .obsidian/ directory; treated as a plain Markdown tree)"
    )
    return Check("config", "ok", f"vault {vault_path} from {source}{note}"), settings


def _projection(settings: Settings | None) -> Check:
    if settings is None:
        return Check("projection", "skip", "no vault configured")
    if not settings.db_path.exists():
        return Check(
            "projection", "fail", f"no projection at {settings.db_path}", "run `llmwiki index`"
        )
    report = dbmod.inspect_integrity(settings.db_path, vault_path=settings.vault_path)
    if not report.get("ok"):
        return Check(
            "projection",
            "fail",
            "integrity problems: "
            + ", ".join(
                k
                for k, v in report.items()
                if v and k.startswith(("orphan", "chunks_without", "documents_missing", "mixed"))
            ),
            "run `llmwiki index --mode full`",
        )
    age = ""
    try:
        with dbmod.connect(settings.db_path) as conn:
            row = conn.execute(
                "SELECT finished_at_ns FROM index_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
            docs = int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        if row and row[0]:
            hours = (time.time_ns() - int(row[0])) / 3.6e12
            age = f", last index {hours:.1f} h ago"
            if hours > 24:
                return Check(
                    "projection",
                    "warn",
                    f"{docs} documents, schema v{report['schema_version']}{age}",
                    "run `llmwiki index` or keep `llmwiki index --watch` running",
                )
        return Check(
            "projection", "ok", f"{docs} documents, schema v{report['schema_version']}{age}"
        )
    except Exception as exc:
        return Check("projection", "warn", f"could not read run history: {type(exc).__name__}")


def _hermes(settings: Settings | None) -> Check:
    home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    if not home.is_dir():
        return Check("hermes", "skip", "no ~/.hermes; Hermes not installed on this machine")
    link = home / "plugins" / "llmwiki"
    if not link.exists():
        return Check(
            "hermes",
            "warn",
            "plugin not linked into ~/.hermes/plugins",
            "ln -s <repo>/hermes_plugin ~/.hermes/plugins/llmwiki && hermes plugins enable llmwiki --no-allow-tool-override",
        )
    enabled = False
    vault_set = False
    try:
        import yaml

        cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}
        plugins = cfg.get("plugins") or {}
        enabled = "llmwiki" in (plugins.get("enabled") or [])
        vault_set = bool(
            ((plugins.get("entries") or {}).get("llmwiki") or {}).get("settings", {}).get("vault")
        )
    except Exception:
        pass
    if not enabled:
        return Check(
            "hermes",
            "warn",
            "plugin linked but not enabled",
            "hermes plugins enable llmwiki --no-allow-tool-override",
        )
    if not vault_set:
        fallback = " (will fall back to the llmwiki user config)" if settings else ""
        return Check(
            "hermes",
            "warn" if settings else "fail",
            f"plugin enabled, settings.vault unset{fallback}",
            f"hermes config set plugins.entries.llmwiki.settings.vault {settings.vault_path if settings else '<vault>'}",
        )
    return Check(
        "hermes", "ok", "plugin linked, enabled, vault set; restart the gateway after changes"
    )


def run_doctor() -> list[Check]:
    config_check, settings = _config()
    return [
        _python(),
        _sqlite(),
        _model(settings),
        config_check,
        _projection(settings),
        _hermes(settings),
    ]


def format_checks(checks: list[Check]) -> str:
    icons = {"ok": "ok  ", "warn": "warn", "fail": "FAIL", "skip": "skip"}
    lines = []
    for c in checks:
        lines.append(f"[{icons[c.status]}] {c.name:10} {c.detail}")
        if c.next_step and c.status in ("warn", "fail"):
            lines.append(f"            -> {c.next_step}")
    return "\n".join(lines)


__all__ = ["Check", "format_checks", "run_doctor"]
