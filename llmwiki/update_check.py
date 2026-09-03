"""Advisory release-update checks for all llmwiki host surfaces.

The check intentionally has no installation capability. It consults PyPI first
and falls back to the project's latest published GitHub Release if PyPI cannot
answer, which supports the pre-publication period as well as normal packages.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from collections.abc import Callable
from typing import Any

from packaging.version import InvalidVersion, Version

PACKAGE_NAME = "hermes-llmwiki-rag"
PYPI_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
GITHUB_LATEST_RELEASE_URL = (
    "https://api.github.com/repos/TechZeph/hermes-llmwiki-rag/releases/latest"
)
FetchJson = Callable[[str, float], dict[str, Any]]
CheckForUpdate = Callable[..., dict[str, str]]


def _fetch_json(url: str, timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": f"{PACKAGE_NAME} update-check",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("release endpoint returned a non-object response")
    return payload


def _newer(candidate: str, current: str) -> bool:
    try:
        return Version(candidate) > Version(current)
    except InvalidVersion:
        return False


def _status(*, state: str, current_version: str, **extra: str) -> dict[str, str]:
    return {"state": state, "current_version": current_version, **extra}


def check_for_update(
    *,
    current_version: str,
    fetch_json: FetchJson = _fetch_json,
    timeout_s: float = 2.0,
) -> dict[str, str]:
    """Return a safe advisory update result, preferring PyPI over GitHub.

    Network, endpoint and version-format failures are represented as an
    ``unavailable`` result; they never prevent a host from starting.
    """
    try:
        pypi = fetch_json(PYPI_URL, timeout_s)
        latest = str(pypi["info"]["version"])
    except (KeyError, OSError, TypeError, ValueError):
        latest = ""
    else:
        return _status(
            state="update_available" if _newer(latest, current_version) else "up_to_date",
            current_version=current_version,
            latest_version=latest,
            source="pypi",
            url=f"https://pypi.org/project/{PACKAGE_NAME}/{latest}/",
        )

    try:
        github = fetch_json(GITHUB_LATEST_RELEASE_URL, timeout_s)
        latest = str(github["tag_name"]).lstrip("v")
        url = str(github["html_url"])
    except (KeyError, OSError, TypeError, ValueError):
        return _status(state="unavailable", current_version=current_version)
    return _status(
        state="update_available" if _newer(latest, current_version) else "up_to_date",
        current_version=current_version,
        latest_version=latest,
        source="github",
        url=url,
    )


class UpdateChecker:
    """Run one bounded, advisory check without delaying host startup."""

    def __init__(
        self,
        *,
        current_version: str,
        check: CheckForUpdate = check_for_update,
        timeout_s: float = 2.0,
    ) -> None:
        self._current_version = current_version
        self._check = check
        self._timeout_s = timeout_s
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._status: dict[str, str] = {
            "state": "not_checked",
            "current_version": current_version,
        }

    def start(self) -> bool:
        """Start exactly one daemon check and report whether this call did so."""
        with self._lock:
            if self._thread is not None:
                return False
            self._status = {"state": "checking", "current_version": self._current_version}
            self._thread = threading.Thread(
                target=self._run, name="llmwiki-update-check", daemon=True
            )
            self._thread.start()
            return True

    def _run(self) -> None:
        try:
            result = self._check(current_version=self._current_version, timeout_s=self._timeout_s)
        except Exception:  # defensive boundary: an advisory check must fail closed
            result = {"state": "unavailable"}
        with self._lock:
            self._status = {"current_version": self._current_version, **result}

    def status(self) -> dict[str, str]:
        with self._lock:
            return dict(self._status)

    def wait(self, timeout_s: float | None = None) -> bool:
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout_s)
        return not thread.is_alive()


__all__ = ["UpdateChecker", "check_for_update"]
