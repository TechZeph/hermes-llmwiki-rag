"""Lexical (BM25) retrieval over SQLite FTS5.

The ``chunks_fts`` virtual table is a projection of ``chunks`` joined with
the parent document title. It is maintained by ``AFTER INSERT`` /
``AFTER DELETE`` triggers on ``chunks`` (see :mod:`llmwiki.db`), so it is
transactionally consistent with the relational rows by construction: a
rolled-back chunk replacement rolls the FTS rows back with it, and a
cascading document delete removes them.

Query construction is deliberately conservative. User text is never
passed to FTS5 as-is; each term is quoted as a phrase so FTS5 operators
(``AND``, ``NOT``, ``*``, ``:``, parentheses) in user input cannot change
the query semantics or raise syntax errors.
"""

from __future__ import annotations

import re
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Final

from .corpus import profile_predicate

FTS_TABLE: Final = "chunks_fts"
FTS_TOKENIZER: Final = "porter unicode61 remove_diacritics 2"

# Column weights for bm25(): text, section_name, heading, title. Heading and
# title carry more signal per token than body text.
FTS_COLUMN_WEIGHTS: Final = (1.0, 2.0, 3.0, 4.0)

# Extremely common English function words that carry no lexical signal and
# would otherwise match nearly every chunk under OR semantics.
_STOPWORDS: Final = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "at",
        "by",
        "from",
        "into",
        "about",
        "over",
        "under",
        "than",
        "then",
        "there",
        "here",
        "we",
        "you",
        "i",
        "me",
        "my",
        "our",
        "your",
        "they",
        "them",
        "their",
        "he",
        "she",
        "his",
        "her",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "not",
        "no",
        "yes",
        "if",
    ]
)

# A token is a run of word characters optionally joined by internal
# punctuation such as ``bge-small-en-v1.5`` or ``pre_llm_call``.
_TOKEN_RE: Final = re.compile(r"[\w][\w\-./+]*[\w]|[\w]", re.UNICODE)


def build_match_query(text: str) -> str:
    """Return a safe FTS5 MATCH expression for free-form user text.

    Every term becomes a quoted phrase (so hyphenated identifiers match as
    an adjacent token sequence), joined with ``OR`` so BM25 ranks partial
    matches instead of requiring every term. Returns an empty string when
    the text has no usable terms; callers must treat that as "no lexical
    candidates" rather than passing it to FTS5.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0).strip("-./+")
        if not token:
            continue
        lowered = token.lower()
        if lowered in _STOPWORDS or lowered in seen:
            continue
        seen.add(lowered)
        terms.append('"' + token.replace('"', "") + '"')
    return " OR ".join(terms)


class LexicalIndex(ABC):
    """Abstract lexical index (BM25 over chunk text)."""

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int,
        *,
        profile: str = "all",
        document_ids: set[int] | None = None,
        only_document_ids: set[int] | None = None,
        updated_after_ns: int | None = None,
        updated_before_ns: int | None = None,
    ) -> list[tuple[int, float]]:
        """Return ``(chunk_id, score)`` pairs, best first; larger score is better.

        ``document_ids`` widens the profile to an explicit document set
        (graph-expanded profiles); ``None`` means the profile predicate alone.
        """

    @abstractmethod
    def count(self) -> int:
        """Number of rows currently indexed."""


class Fts5Index(LexicalIndex):
    """BM25 search over the trigger-maintained ``chunks_fts`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def count(self) -> int:
        row = self._conn.execute(f"SELECT COUNT(*) FROM {FTS_TABLE}").fetchone()
        return int(row[0]) if row else 0

    def search(
        self,
        query: str,
        top_k: int,
        *,
        profile: str = "all",
        document_ids: set[int] | None = None,
        only_document_ids: set[int] | None = None,
        updated_after_ns: int | None = None,
        updated_before_ns: int | None = None,
    ) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        match = build_match_query(query)
        if not match:
            return []
        predicate, params = profile_predicate(profile, alias="d")
        if document_ids:
            ids = sorted(document_ids)[:900]
            predicate = f"({predicate} OR d.id IN ({','.join('?' * len(ids))}))"
            params = [*params, *ids]
        if only_document_ids is not None:
            if not only_document_ids:
                return []
            ids = sorted(only_document_ids)[:900]
            predicate = f"({predicate} AND d.id IN ({','.join('?' * len(ids))}))"
            params = [*params, *ids]
        if updated_after_ns is not None:
            predicate = f"({predicate} AND d.updated_at_ns >= ?)"
            params = [*params, int(updated_after_ns)]
        if updated_before_ns is not None:
            predicate = f"({predicate} AND d.updated_at_ns <= ?)"
            params = [*params, int(updated_before_ns)]
        weights = ", ".join(str(w) for w in FTS_COLUMN_WEIGHTS)
        rows = self._conn.execute(
            f"""
            SELECT f.rowid, bm25({FTS_TABLE}, {weights}) AS score
            FROM {FTS_TABLE} f
            JOIN chunks c ON c.id = f.rowid
            JOIN documents d ON d.id = c.document_id
            WHERE {FTS_TABLE} MATCH ? AND {predicate}
            ORDER BY score
            LIMIT ?
            """,
            (match, *params, top_k),
        ).fetchall()
        # FTS5's bm25() is "smaller is better" (negative). Negate so the
        # public contract is "larger is better" like every other relevance
        # score, while keeping the value's magnitude meaningful.
        return [(int(chunk_id), -float(score)) for chunk_id, score in rows]

    def rebuild(self) -> int:
        """Repopulate the FTS projection from ``chunks``; returns the row count.

        Used by migrations and by integrity repair. Runs inside the
        caller's transaction when one is open.
        """
        self._conn.execute(f"DELETE FROM {FTS_TABLE}")
        self._conn.execute(
            f"""
            INSERT INTO {FTS_TABLE}(rowid, text, section_name, heading, title)
            SELECT c.id, c.text, c.section_name,
                   (SELECT group_concat(value, ' ') FROM json_each(c.heading_path_json)),
                   d.title
            FROM chunks c JOIN documents d ON d.id = c.document_id
            """
        )
        return self.count()


def fts_terms(texts: Sequence[str]) -> list[str]:
    """Debug helper: the phrase terms ``build_match_query`` would emit."""
    return [build_match_query(t) for t in texts]


__all__ = [
    "FTS_COLUMN_WEIGHTS",
    "FTS_TABLE",
    "FTS_TOKENIZER",
    "Fts5Index",
    "LexicalIndex",
    "build_match_query",
    "fts_terms",
]
