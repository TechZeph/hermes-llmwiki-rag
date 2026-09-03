"""Release-update discovery is advisory and never installs anything."""

from __future__ import annotations

import threading
from typing import Any

from llmwiki.update_check import UpdateChecker, check_for_update


def test_check_prefers_pypi_when_it_reports_a_newer_release() -> None:
    calls: list[str] = []

    def fetch_json(url: str, timeout_s: float) -> dict[str, Any]:
        calls.append(url)
        assert timeout_s == 2.0
        return {"info": {"version": "0.2.0"}}

    status = check_for_update(current_version="0.1.0", fetch_json=fetch_json, timeout_s=2.0)

    assert status["state"] == "update_available"
    assert status["source"] == "pypi"
    assert status["current_version"] == "0.1.0"
    assert status["latest_version"] == "0.2.0"
    assert len(calls) == 1 and "pypi.org" in calls[0]


def test_check_falls_back_to_github_releases_when_pypi_is_unavailable() -> None:
    calls: list[str] = []

    def fetch_json(url: str, timeout_s: float) -> dict[str, Any]:
        calls.append(url)
        if "pypi.org" in url:
            raise OSError("offline mirror")
        return {
            "tag_name": "v0.2.0",
            "html_url": "https://github.com/TechZeph/hermes-llmwiki-rag/releases/tag/v0.2.0",
        }

    status = check_for_update(current_version="0.1.0", fetch_json=fetch_json)

    assert status["state"] == "update_available"
    assert status["source"] == "github"
    assert status["latest_version"] == "0.2.0"
    assert status["url"].endswith("/v0.2.0")
    assert len(calls) == 2


def test_background_checker_reports_checking_then_the_advisory_result() -> None:
    started = threading.Event()
    release = threading.Event()

    def check(*, current_version: str, timeout_s: float) -> dict[str, str]:
        assert current_version == "0.1.0" and timeout_s == 2.0
        started.set()
        assert release.wait(1)
        return {"state": "update_available", "latest_version": "0.2.0", "source": "pypi"}

    checker = UpdateChecker(current_version="0.1.0", check=check, timeout_s=2.0)
    checker.start()
    assert started.wait(1)
    assert checker.status()["state"] == "checking"
    assert checker.start() is False

    release.set()
    assert checker.wait(1)
    assert checker.status() == {
        "state": "update_available",
        "current_version": "0.1.0",
        "latest_version": "0.2.0",
        "source": "pypi",
    }
