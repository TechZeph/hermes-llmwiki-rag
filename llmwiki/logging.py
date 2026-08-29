"""Structured logging for the RAG core.

The RAG is used in three contexts:

1. From the CLI (human-readable, single-line).
2. From the Hermes plugin (JSON to stdout, picked up by Hermes logging).
3. From the evaluation harness (one block per retrieval run).

The `setup_logging` function picks the right format based on the
`LLMWIKI_LOG_FORMAT` environment variable. Default is "text".

The RAG core never configures handlers at import time; the entry point
(CLI / plugin / test) calls `setup_logging` explicitly. This keeps the
library importable without side effects.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final, TextIO

_LOG_FORMAT_TEXT: Final = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_LOG_FORMAT_JSON: Final = (
    '{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
)

_TEXT = "text"
_JSON = "json"
_VALID_FORMATS: Final = frozenset({_TEXT, _JSON})

_DEFAULT_LEVEL: Final = "INFO"


def setup_logging(
    level: str | None = None,
    fmt: str | None = None,
    *,
    stream: TextIO | None = None,
) -> None:
    """Configure the root logger for the `llmwiki` package.

    Parameters
    ----------
    level:
        Log level name (e.g. "DEBUG", "INFO"). Defaults to the
        ``LLMWIKI_LOG_LEVEL`` env var, or "INFO".
    fmt:
        Output format, either "text" or "json". Defaults to the
        ``LLMWIKI_LOG_FORMAT`` env var, or "text".
    stream:
        Output stream; defaults to ``sys.stderr``. Mostly useful in
        tests to capture output.
    """
    resolved_level = (level or os.environ.get("LLMWIKI_LOG_LEVEL") or _DEFAULT_LEVEL).upper()
    resolved_fmt = (fmt or os.environ.get("LLMWIKI_LOG_FORMAT") or _TEXT).lower()
    if resolved_fmt not in _VALID_FORMATS:
        raise ValueError(
            f"invalid log format {resolved_fmt!r}; expected one of {sorted(_VALID_FORMATS)}"
        )

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(
        logging.Formatter(_LOG_FORMAT_JSON if resolved_fmt == _JSON else _LOG_FORMAT_TEXT)
    )

    root = logging.getLogger("llmwiki")
    # Idempotent: clear any handlers we previously added so repeated
    # calls (e.g. in tests) don't duplicate output.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved_level)
    # Don't propagate to Python's root logger; we own the log surface.
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the ``llmwiki`` package logger."""
    return logging.getLogger(f"llmwiki.{name}")
