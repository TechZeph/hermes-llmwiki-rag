"""Portable, advisory resource controls for local embedding batches.

These controls decide whether to begin another batch; they are not operating
system memory limits. Callers that need an enforceable Linux limit should run
in a dedicated systemd scope with ``MemoryHigh=`` / ``MemoryMax=``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Settings


_PROFILE_DEFAULTS: dict[str, tuple[int, int]] = {
    "conservative": (8, 2),
    "balanced": (32, 4),
    "performance": (128, max(1, (os.cpu_count() or 1) // 2)),
}


class ResourceBudgetExceeded(RuntimeError):
    """Raised before a batch when local memory headroom is insufficient."""


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Low-overhead memory view in MiB; unavailable metrics are ``None``."""

    rss_mb: int | None
    available_mb: int | None
    cgroup_remaining_mb: int | None = None


def _read_cgroup_remaining_mb() -> int | None:
    """Return local cgroup-v2 memory headroom when it is readable."""
    if os.name != "posix":
        return None
    root = "/sys/fs/cgroup"
    try:
        with (
            open(f"{root}/memory.max", encoding="utf-8") as max_f,
            open(f"{root}/memory.current", encoding="utf-8") as cur_f,
        ):
            maximum = max_f.read().strip()
            current = int(cur_f.read().strip())
        if maximum == "max":
            return None
        return max(0, (int(maximum) - current) // (1024 * 1024))
    except (OSError, ValueError):
        return None


def read_resource_snapshot() -> ResourceSnapshot:
    """Read portable process/system memory plus cgroup-v2 headroom when present."""
    cgroup_remaining = _read_cgroup_remaining_mb()
    rss_mb: int | None = None
    available_mb: int | None = None
    try:
        import psutil

        process = psutil.Process()
        rss_mb = int(process.memory_info().rss // (1024 * 1024))
        available_mb = int(psutil.virtual_memory().available // (1024 * 1024))
    except Exception:
        pass
    if available_mb is not None and cgroup_remaining is not None:
        available_mb = min(available_mb, cgroup_remaining)
    return ResourceSnapshot(
        rss_mb=rss_mb,
        available_mb=available_mb,
        cgroup_remaining_mb=cgroup_remaining,
    )


class EmbeddingBatchController:
    """Choose bounded batches and refuse work that would cross configured floors."""

    def __init__(
        self,
        settings: Settings,
        *,
        snapshot: Callable[[], ResourceSnapshot] = read_resource_snapshot,
    ) -> None:
        try:
            default_batch, default_threads = _PROFILE_DEFAULTS[settings.resource_profile]
        except KeyError as exc:
            raise ValueError(f"unknown resource profile: {settings.resource_profile!r}") from exc
        self._snapshot = snapshot
        self.effective_batch_size = settings.embedding_batch_size or default_batch
        self.effective_threads = settings.embedding_threads or default_threads
        self._minimum_batch_size = settings.embedding_min_batch_size
        self._memory_budget_mb = settings.embedding_memory_budget_mb
        self._minimum_available_mb = settings.embedding_min_available_mb
        self._previous: ResourceSnapshot | None = None

    def next_batch_size(self, remaining: int) -> int:
        """Admit and size the next batch without beginning native model work."""
        if remaining < 1:
            return 0
        current = self._snapshot()
        if (
            self._memory_budget_mb is not None
            and current.rss_mb is not None
            and current.rss_mb > self._memory_budget_mb
        ):
            raise ResourceBudgetExceeded(
                f"embedding RSS {current.rss_mb} MiB exceeds budget {self._memory_budget_mb} MiB"
            )
        if current.available_mb is not None and current.available_mb < self._minimum_available_mb:
            raise ResourceBudgetExceeded(
                f"available memory {current.available_mb} MiB is below floor "
                f"{self._minimum_available_mb} MiB"
            )
        self._previous = current
        return min(remaining, self.effective_batch_size)

    def record_batch_complete(self) -> ResourceSnapshot:
        """Record pressure after persistence and reduce later batches on a large RSS jump."""
        current = self._snapshot()
        previous = self._previous
        if (
            previous is not None
            and previous.rss_mb is not None
            and current.rss_mb is not None
            and current.rss_mb > previous.rss_mb * 1.25
        ):
            self.effective_batch_size = max(
                self._minimum_batch_size, self.effective_batch_size // 2
            )
        self._previous = current
        return current

    def status(self) -> dict[str, int | str | None]:
        """Return safe diagnostics for run logs and host status responses."""
        snapshot = self._snapshot()
        return {
            "batch_size": self.effective_batch_size,
            "threads": self.effective_threads,
            "memory_budget_mb": self._memory_budget_mb,
            "minimum_available_mb": self._minimum_available_mb,
            "rss_mb": snapshot.rss_mb,
            "available_mb": snapshot.available_mb,
            "cgroup_remaining_mb": snapshot.cgroup_remaining_mb,
            "enforcement": "advisory",
        }


__all__ = [
    "EmbeddingBatchController",
    "ResourceBudgetExceeded",
    "ResourceSnapshot",
    "read_resource_snapshot",
]
