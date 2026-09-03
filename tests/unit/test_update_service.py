"""Service-level startup update-check behaviour."""

from __future__ import annotations

from pathlib import Path

from llmwiki.service import ServiceConfig, WikiService
from llmwiki.update_check import UpdateChecker


def test_service_starts_update_check_once_and_exposes_result(tmp_path: Path) -> None:
    checker = UpdateChecker(
        current_version="0.1.0",
        check=lambda **_: {
            "state": "update_available",
            "latest_version": "0.2.0",
            "source": "pypi",
        },
    )
    service = WikiService(ServiceConfig(vault=str(tmp_path)), update_checker=checker)

    assert service.status()["update_check"]["state"] == "not_checked"
    assert service.start_update_check() is True
    assert service.start_update_check() is False
    assert checker.wait(1)
    assert service.status()["update_check"] == {
        "state": "update_available",
        "current_version": "0.1.0",
        "latest_version": "0.2.0",
        "source": "pypi",
    }
    service.close()
