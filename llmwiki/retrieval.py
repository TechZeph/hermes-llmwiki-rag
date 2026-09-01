"""End-to-end retrieval: profile → channels → fusion → policy → candidates.

The pipeline follows the architecture contract:

1. The caller names a corpus profile (``answer``, ``project:<id>``,
   ``evidence``, ``history``, ``all``). Profiles constrain candidates
   before ranking; they are not a ranking signal.
2. Dense (sqlite-vec) and lexical (FTS5 BM25) candidates are retrieved
   independently with their raw metrics preserved.
3. Reciprocal-rank fusion orders the union (``hybrid`` mode) or a
   single channel is used as-is (``dense`` / ``lexical``).
4. Optional reranking re-scores the top fused candidates when enabled.
5. Authority policy and document diversification are applied as a
   stable re-ordering; scores are never mixed across channels.

Nothing here imports Hermes; the plugin is a thin adapter over
:class:`Retriever`.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace

from .authority import apply_authority_policy, detect_intent
from .citations import ContextBlock, build_context
from .config import Settings
from .corpus import profile_matches, profile_predicate
from .embeddings import Embedder
from .hybrid import diversify, reciprocal_rank_fusion
from .lexical import Fts5Index, LexicalIndex
from .models import Candidate, RetrievalResult
from .recipes import format_query_embedding_input
from .reranker import Reranker
from .vector import SqliteVecStore, VectorStore

VALID_MODES = ("dense", "lexical", "hybrid")


def hydrate_candidates(conn: sqlite3.Connection, chunk_ids: Sequence[int]) -> dict[int, Candidate]:
    """Load typed candidates (without channel metrics) for ``chunk_ids``."""
    if not chunk_ids:
        return {}
    out: dict[int, Candidate] = {}
    ids = [int(i) for i in chunk_ids]
    for start in range(0, len(ids), 500):
        batch = ids[start : start + 500]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"""
            SELECT c.id, c.document_id, d.path, d.title, c.heading_path_json,
                   c.section_name, c.position, c.text, c.text_hash,
                   d.source_kind, d.page_role, d.project_id, d.updated_at_ns, d.is_route_map
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({placeholders})
            """,
            batch,
        ).fetchall()
        for row in rows:
            out[int(row[0])] = Candidate(
                chunk_id=int(row[0]),
                document_id=int(row[1]),
                path=str(row[2]),
                title=str(row[3]),
                heading_path=tuple(str(h) for h in json.loads(str(row[4]))),
                section_name=str(row[5]),
                position=int(row[6]),
                text=str(row[7]),
                text_hash=str(row[8]),
                source_kind=str(row[9]),
                page_role=str(row[10]),
                project_id=str(row[11]) if row[11] is not None else None,
                updated_at_ns=int(row[12]),
                is_route_map=bool(row[13]),
            )
    return out


def _profile_metadata(
    conn: sqlite3.Connection, chunk_ids: Sequence[int]
) -> dict[int, dict[str, object]]:
    if not chunk_ids:
        return {}
    out: dict[int, dict[str, object]] = {}
    ids = [int(i) for i in chunk_ids]
    for start in range(0, len(ids), 500):
        batch = ids[start : start + 500]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"""
            SELECT c.id, d.source_kind, d.page_role, d.project_id, d.is_route_map
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({placeholders})
            """,
            batch,
        ).fetchall()
        for chunk_id, source_kind, page_role, project_id, is_route_map in rows:
            out[int(chunk_id)] = {
                "source_kind": str(source_kind),
                "page_role": str(page_role),
                "project_id": str(project_id) if project_id is not None else None,
                "is_route_map": bool(is_route_map),
            }
    return out


class Retriever:
    """Profile-aware hybrid retriever over one open projection connection.

    Parameters
    ----------
    conn:
        Open connection from :func:`llmwiki.db.connect` with the schema
        already initialised.
    embedder:
        Query embedder; required for ``dense`` and ``hybrid`` modes.
    settings:
        Retrieval knobs (candidate depths, ``rrf_k``, diversification cap,
        reranker policy). Per-call arguments override them.
    vector_store / lexical_index / reranker:
        Injectable for tests and experiments.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        embedder: Embedder | None,
        settings: Settings,
        vector_store: VectorStore | None = None,
        lexical_index: LexicalIndex | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._conn = conn
        self._embedder = embedder
        self._settings = settings
        self._vectors = vector_store or SqliteVecStore(conn)
        self._lexical = lexical_index or Fts5Index(conn)
        self._reranker = reranker

    # --- channels -----------------------------------------------------------

    def dense_channel(self, query: str, *, profile: str, top_k: int) -> list[tuple[int, float]]:
        """Profile-filtered nearest neighbours as ``(chunk_id, distance)``."""
        if self._embedder is None:
            raise ValueError("dense retrieval requires an embedder")
        if top_k <= 0:
            return []
        total = self._vectors.count()
        if total == 0:
            return []
        q_vec = self._embedder.embed(
            [format_query_embedding_input(query, recipe=self._settings.query_recipe)]
        )[0]
        # The vec0 table cannot filter on document metadata inside the KNN
        # query, so over-fetch and filter. Widen progressively rather than
        # always scanning the whole store.
        fetch = min(total, max(top_k * 4, 64))
        while True:
            hits = self._vectors.search(q_vec, top_k=fetch)
            meta = _profile_metadata(self._conn, [cid for cid, _ in hits])
            kept = [
                (cid, dist)
                for cid, dist in hits
                if (m := meta.get(cid)) is not None and profile_matches(profile, m)
            ]
            if len(kept) >= top_k or fetch >= total or len(hits) < fetch:
                return kept[:top_k]
            fetch = min(total, fetch * 4)

    def lexical_channel(self, query: str, *, profile: str, top_k: int) -> list[tuple[int, float]]:
        """Profile-filtered BM25 hits as ``(chunk_id, score)``, larger is better."""
        profile_predicate(profile)  # validate early with the same error text
        return self._lexical.search(query, top_k, profile=profile)

    # --- pipeline -----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        profile: str = "answer",
        mode: str | None = None,
        top_k: int | None = None,
        top_k_dense: int | None = None,
        top_k_lexical: int | None = None,
        rrf_k: int | None = None,
        rrf_weights: Mapping[str, float] | None = None,
        max_per_document: int | None = None,
        rerank: bool | None = None,
        apply_authority: bool = True,
    ) -> RetrievalResult:
        started = time.perf_counter()
        s = self._settings
        mode = mode or s.retrieval_mode
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        profile_predicate(profile)
        final_k = top_k if top_k is not None else s.retrieval_top_k_final
        k_dense = top_k_dense if top_k_dense is not None else s.retrieval_top_k_dense
        k_lex = top_k_lexical if top_k_lexical is not None else s.retrieval_top_k_lexical
        k_rrf = rrf_k if rrf_k is not None else s.rrf_k
        per_doc = max_per_document if max_per_document is not None else s.max_chunks_per_document
        use_reranker = rerank if rerank is not None else s.reranker_enabled
        query = query.strip()
        if not query:
            return RetrievalResult(query=query, profile=profile, mode=mode, candidates=())

        dense_hits: list[tuple[int, float]] = []
        lexical_hits: list[tuple[int, float]] = []
        if mode in ("dense", "hybrid"):
            dense_hits = self.dense_channel(query, profile=profile, top_k=k_dense)
        if mode in ("lexical", "hybrid"):
            lexical_hits = self.lexical_channel(query, profile=profile, top_k=k_lex)

        dense_rank = {cid: i for i, (cid, _) in enumerate(dense_hits, start=1)}
        dense_dist = {cid: d for cid, d in dense_hits}
        lex_rank = {cid: i for i, (cid, _) in enumerate(lexical_hits, start=1)}
        lex_score = {cid: sc for cid, sc in lexical_hits}

        if mode == "hybrid":
            fused = reciprocal_rank_fusion(
                {"dense": [c for c, _ in dense_hits], "lexical": [c for c, _ in lexical_hits]},
                k=k_rrf,
                weights=rrf_weights
                or {"dense": s.rrf_dense_weight, "lexical": s.rrf_lexical_weight},
            )
            ordered = [e.id for e in fused]
            rrf = {e.id: e.rrf_score for e in fused}
        elif mode == "dense":
            ordered = [c for c, _ in dense_hits]
            rrf = {}
        else:
            ordered = [c for c, _ in lexical_hits]
            rrf = {}

        hydrated = hydrate_candidates(self._conn, ordered)
        candidates: list[Candidate] = []
        for cid in ordered:
            base = hydrated.get(cid)
            if base is None:
                continue  # raced with a delete; skip rather than fail
            candidates.append(
                replace(
                    base,
                    dense_rank=dense_rank.get(cid),
                    dense_distance=dense_dist.get(cid),
                    lexical_rank=lex_rank.get(cid),
                    bm25_score=lex_score.get(cid),
                    rrf_score=rrf.get(cid),
                    selection_reason=mode,
                )
            )

        # Reranking is a relevance signal over the fused head; the authority
        # policy runs afterwards so it always has the final, inspectable say.
        if use_reranker and self._reranker is not None and candidates:
            candidates = self._rerank(query, candidates, limit=s.rerank_candidates)

        intent = detect_intent(query)
        conflicts: tuple[str, ...] = ()
        if apply_authority:
            candidates, conflicts = apply_authority_policy(
                candidates, intent=intent, profile=profile
            )

        group_of = {c.chunk_id: c.document_id for c in candidates}
        kept_ids = diversify([c.chunk_id for c in candidates], group_of, max_per_group=per_doc)
        by_id = {c.chunk_id: c for c in candidates}
        final = [by_id[cid] for cid in kept_ids][:final_k]

        return RetrievalResult(
            query=query,
            profile=profile,
            mode=mode,
            candidates=tuple(final),
            dense_returned=len(dense_hits),
            lexical_returned=len(lexical_hits),
            fused_total=len(ordered),
            intent=intent,
            conflicts=conflicts,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _rerank(self, query: str, candidates: list[Candidate], *, limit: int) -> list[Candidate]:
        assert self._reranker is not None
        head = candidates[: max(limit, 1)]
        tail = candidates[len(head) :]
        scored = self._reranker.rerank(query, [c.text for c in head])
        reordered = [
            replace(
                head[idx],
                rerank_score=score,
                selection_reason=f"{head[idx].selection_reason}+rerank",
            )
            for idx, score in scored
        ]
        return reordered + tail


def context_for(result: RetrievalResult, settings: Settings) -> ContextBlock:
    """Render a retrieval result with the configured context budgets."""
    return build_context(
        result.candidates,
        conflicts=result.conflicts,
        total_budget_tokens=settings.context_budget_tokens,
        per_document_budget_tokens=settings.context_per_document_tokens,
        max_excerpts=settings.context_max_excerpts,
        retrieval_mode=result.mode,
    )


def describe_channels(result: RetrievalResult) -> Mapping[str, int]:
    """Small summary used by the CLI and status surfaces."""
    return {
        "dense": result.dense_returned,
        "lexical": result.lexical_returned,
        "fused": result.fused_total,
        "returned": len(result.candidates),
    }


__all__ = ["VALID_MODES", "Retriever", "context_for", "describe_channels", "hydrate_candidates"]
