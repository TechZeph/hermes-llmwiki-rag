"""Small cross-platform process/system helpers (no POSIX-only imports at module level)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_IS_POSIX = os.name == "posix"
_IS_WINDOWS = os.name == "nt"


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB; 0.0 when unavailable."""
    if _IS_POSIX:
        try:
            import resource

            kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports kilobytes, macOS bytes.
            return round(kb / (1024.0 * 1024.0) if sys.platform == "darwin" else kb / 1024.0, 1)
        except Exception:
            return 0.0
    if _IS_WINDOWS:
        try:
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _Counters()
            counters.cb = ctypes.sizeof(_Counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
            if ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
                handle, ctypes.byref(counters), counters.cb
            ):
                return round(float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0), 1)
        except Exception:
            return 0.0
    return 0.0


def user_config_dir() -> Path:
    """Per-user config directory: XDG on POSIX, %APPDATA% on Windows."""
    override = os.environ.get("XDG_CONFIG_HOME")
    if override:
        return Path(override) / "llmwiki"
    if _IS_WINDOWS:
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "llmwiki"
    return Path.home() / ".config" / "llmwiki"


def user_data_dir() -> Path:
    """Per-user data directory: XDG on POSIX, %LOCALAPPDATA% on Windows."""
    override = os.environ.get("XDG_DATA_HOME")
    if override:
        return Path(override) / "llmwiki"
    if _IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "llmwiki"
    return Path.home() / ".local" / "share" / "llmwiki"


__all__ = ["peak_rss_mb", "user_config_dir", "user_data_dir"]
