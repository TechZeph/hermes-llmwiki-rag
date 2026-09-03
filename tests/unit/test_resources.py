"""Resource-policy tests for bounded local embedding work."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmwiki.config import Settings
from llmwiki.resources import EmbeddingBatchController, ResourceBudgetExceeded, ResourceSnapshot


def _settings(**changes: object) -> Settings:
    values: dict[str, object] = {
        "vault_path": Path("/vault"),
        "db_path": Path("/db.sqlite"),
    }
    values.update(changes)
    return Settings(**values)  # type: ignore[arg-type]


def test_conservative_profile_uses_small_single_worker_batches() -> None:
    controller = EmbeddingBatchController(
        _settings(resource_profile="conservative"),
        snapshot=lambda: ResourceSnapshot(rss_mb=100, available_mb=4096),
    )

    assert controller.effective_batch_size == 8
    assert controller.effective_threads == 2
    assert controller.next_batch_size(100) == 8


def test_explicit_batch_and_thread_overrides_win_over_profile() -> None:
    controller = EmbeddingBatchController(
        _settings(resource_profile="conservative", embedding_batch_size=24, embedding_threads=6),
        snapshot=lambda: ResourceSnapshot(rss_mb=100, available_mb=4096),
    )

    assert controller.effective_batch_size == 24
    assert controller.effective_threads == 6


def test_resource_budget_refuses_next_embedding_batch() -> None:
    controller = EmbeddingBatchController(
        _settings(embedding_memory_budget_mb=256, embedding_min_available_mb=1024),
        snapshot=lambda: ResourceSnapshot(rss_mb=257, available_mb=4096),
    )

    with pytest.raises(ResourceBudgetExceeded, match="RSS"):
        controller.next_batch_size(10)


def test_low_available_memory_refuses_next_embedding_batch() -> None:
    controller = EmbeddingBatchController(
        _settings(embedding_min_available_mb=1024),
        snapshot=lambda: ResourceSnapshot(rss_mb=100, available_mb=1023),
    )

    with pytest.raises(ResourceBudgetExceeded, match="available memory"):
        controller.next_batch_size(10)


def test_large_rss_growth_halves_the_following_batch() -> None:
    snapshots = iter(
        [
            ResourceSnapshot(rss_mb=100, available_mb=4096),
            ResourceSnapshot(rss_mb=140, available_mb=4056),
            ResourceSnapshot(rss_mb=140, available_mb=4056),
        ]
    )
    controller = EmbeddingBatchController(
        _settings(resource_profile="balanced"),
        snapshot=lambda: next(snapshots),
    )

    assert controller.next_batch_size(100) == 32
    controller.record_batch_complete()
    assert controller.next_batch_size(100) == 16
