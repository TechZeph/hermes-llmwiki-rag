"""First-run setup: find or create a vault, persist config, run the first index.

Used by ``llmwiki init`` and by the Hermes ``/llmwiki setup`` command. Pure
functions here; the CLI adds the prompts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

STARTER_PAGES: Final[dict[str, str]] = {
    "wiki/index.md": (
        "# Wiki index\n\n**Summary**: Entry point for this wiki.\n\n**Sources**: N/A\n\n"
        "**Last updated**: today\n\n---\n\n- [[welcome]] — how this vault is laid out\n"
        "- [[projects/index]] — project workspaces\n"
    ),
    "wiki/welcome.md": (
        "# Welcome\n\n**Summary**: A starter page explaining the layout llmwiki expects.\n\n"
        "**Sources**: N/A\n\n**Last updated**: today\n\n---\n\n"
        "## Layout\n\n"
        "- `wiki/` holds curated pages: the default `answer` profile searches these.\n"
        "- `wiki/projects/<name>/` holds one workspace per project; `current-state.md` answers "
        "status questions, `decisions.md` answers why, `log.md` is append-only history.\n"
        "- `raw/` and `Clippings/` hold immutable sources; search them with the `evidence` profile.\n"
        "- `wiki/log.md` and every project `log.md` are searched with the `history` profile.\n\n"
        "## Next steps\n\n"
        "Write pages, then run `llmwiki index` (or keep `llmwiki index --watch` running) and ask "
        '`llmwiki search --query "..."`.\n'
    ),
    "wiki/log.md": (
        "# Log\n\n**Summary**: Append-only record of vault changes.\n\n**Sources**: N/A\n\n"
        "**Last updated**: today\n\n---\n\n## [today] create | vault initialised\n- Created by `llmwiki init`.\n"
    ),
    "wiki/projects/index.md": (
        "# Project index\n\n**Summary**: One workspace per project.\n\n**Sources**: N/A\n\n"
        "**Last updated**: today\n\n---\n\n(no projects yet)\n"
    ),
    "raw/README.md": "# raw\n\nImmutable source material (papers, articles, transcripts). Never edited.\n",
    "Clippings/README.md": "# Clippings\n\nHuman-collected clippings and `ideas/` drops. Never edited.\n",
}

_CANDIDATE_ROOTS: Final = (
    "Documents",
    "Obsidian",
    "Workspace",
    "vaults",
    "Vaults",
    "notes",
    "Notes",
    "",
)
_MAX_DEPTH: Final = 3


@dataclass(frozen=True, slots=True)
class VaultCandidate:
    path: Path
    markdown_files: int
    has_obsidian_dir: bool


def looks_like_vault(path: Path) -> bool:
    return path.is_dir() and ((path / ".obsidian").is_dir() or (path / "wiki").is_dir())


def discover_vaults(home: Path | None = None, *, limit: int = 12) -> list[VaultCandidate]:
    """Find directories that look like Obsidian vaults under common roots (bounded scan)."""
    home = home or Path.home()
    found: dict[Path, VaultCandidate] = {}
    for root_name in _CANDIDATE_ROOTS:
        root = home / root_name if root_name else home
        if not root.is_dir():
            continue
        for dirpath, dirnames, _files in os.walk(root):
            current = Path(dirpath)
            depth = len(current.relative_to(root).parts)
            dirnames[:] = [
                d
                for d in dirnames
                if not d.startswith(".")
                and d not in {"node_modules", "venv", ".venv", "site-packages", "__pycache__"}
            ]
            if depth >= _MAX_DEPTH:
                dirnames[:] = []
            if (current / ".obsidian").is_dir() and current not in found:
                md = sum(1 for _ in current.rglob("*.md"))
                found[current] = VaultCandidate(current, md, True)
                dirnames[:] = []  # do not descend into a vault
            if len(found) >= limit:
                break
        if len(found) >= limit:
            break
    return sorted(found.values(), key=lambda c: (-c.markdown_files, str(c.path)))


def create_starter_vault(path: Path) -> Path:
    """Create the minimal layout llmwiki expects; refuses to overwrite existing pages."""
    path = path.expanduser()
    if path.exists() and any(path.iterdir()):
        if looks_like_vault(path):
            raise FileExistsError(
                f"{path} already looks like a vault; pass it to `llmwiki init` instead"
            )
        raise FileExistsError(f"{path} exists and is not empty")
    (path / ".obsidian").mkdir(parents=True, exist_ok=True)
    (path / "Clippings" / "ideas").mkdir(parents=True, exist_ok=True)
    (path / "raw").mkdir(parents=True, exist_ok=True)
    (path / "wiki" / "projects").mkdir(parents=True, exist_ok=True)
    from datetime import date

    today = date.today().isoformat()
    for rel, content in STARTER_PAGES.items():
        target = path / rel
        if not target.exists():
            target.write_text(content.replace("today", today), encoding="utf-8")
    return path


OBSIDIAN_DOWNLOAD_URL: Final = "https://obsidian.md/download"


def find_obsidian() -> str | None:
    """Return how Obsidian can be launched on this machine, or ``None``.

    Obsidian is optional for llmwiki (it reads plain Markdown); this only
    lets ``init`` offer to open a freshly created vault.
    """
    import shutil
    import sys

    for name in ("obsidian", "Obsidian"):
        found = shutil.which(name)
        if found:
            return found
    if sys.platform == "darwin" and Path("/Applications/Obsidian.app").exists():
        return "open -a Obsidian"
    if Path("/snap/bin/obsidian").exists():
        return "/snap/bin/obsidian"
    flatpak = shutil.which("flatpak")
    if flatpak:
        for base in (Path.home() / ".local/share/flatpak", Path("/var/lib/flatpak")):
            if (base / "app" / "md.obsidian.Obsidian").exists():
                return f"{flatpak} run md.obsidian.Obsidian"
    for candidate in sorted((Path.home() / "Applications").glob("Obsidian*.AppImage")):
        if os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def obsidian_open_command(launcher: str, vault: Path) -> list[str]:
    """Command that opens ``vault`` in Obsidian via its ``obsidian://`` URI."""
    from urllib.parse import quote

    uri = "obsidian://open?path=" + quote(str(vault.resolve()), safe="/")
    if launcher == "open -a Obsidian":
        return ["open", "-a", "Obsidian", uri]
    return [*launcher.split(), uri]


def open_in_obsidian(vault: Path, *, launcher: str | None = None) -> bool:
    """Launch Obsidian on ``vault`` in the background; False when unavailable."""
    import subprocess

    launcher = launcher or find_obsidian()
    if not launcher:
        return False
    try:
        subprocess.Popen(
            obsidian_open_command(launcher, vault),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except OSError:
        return False


def estimate_first_index(vault: Path) -> tuple[int, int]:
    """Return ``(markdown_files, estimated_minutes)`` for a cold first index."""
    files = sum(1 for _ in vault.rglob("*.md"))
    # ~10 chunks per page, ~2 chunks/s on a 16-core CPU with the BGE-small model.
    minutes = max(1, round(files * 10 / 2 / 60))
    return files, minutes


__all__ = [
    "OBSIDIAN_DOWNLOAD_URL",
    "STARTER_PAGES",
    "VaultCandidate",
    "create_starter_vault",
    "discover_vaults",
    "estimate_first_index",
    "find_obsidian",
    "looks_like_vault",
    "obsidian_open_command",
    "open_in_obsidian",
]
