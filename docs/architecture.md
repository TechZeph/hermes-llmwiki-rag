# Architecture and delivery contract

This document is the repository-facing architecture for `hermes-llmwiki-rag`. The canonical project planning is maintained in the Clanker vault under `wiki/projects/hermes-llmwiki-rag/`; this file records the contract needed to implement and review the package without depending on that private vault.

## Product boundary

- Markdown is canonical, human-readable knowledge.
- SQLite, FTS5, sqlite-vec, graph edges, and evaluation records are rebuildable projections.
- The package is read-only against source Markdown.
- The retrieval core does not import Hermes.
- Hermes integration is a thin standalone plugin.
- V1 ends at evaluated, authority-aware explicit tools. Automatic injection is opt-in V1.1.

## Corpus profiles and authority

Discovery and retrieval are different concerns. Retrieval always names a profile:

- `answer`: curated wiki pages, excluding append-only history by default.
- `project:<id>`: one project workspace plus linked canonical pages.
- `evidence`: explicit raw/clipping source lookup.
- `history`: explicit append-only log/history lookup.
- `all`: diagnostics only.

Persist deterministic, versioned `source_kind`, `page_role`, `project_id`, `updated_at`, and `is_route_map` metadata. Authority depends on intent: current-state pages answer current-state questions; decisions answer rationale; curated pages answer general facts; raw sources support evidence; logs answer chronology. Return conflicts with provenance rather than silently resolving them by score.

## Projection integrity

Use ordered transactional migrations. A document change must atomically replace its metadata, chunks, vectors, and FTS rows. Explicitly delete rows in virtual tables that cannot enforce foreign keys. Provide:

- a true full rebuild;
- an integrity checker for orphan/stale/mixed-state rows;
- rollback and interrupted-migration tests;
- resolved-path containment inside the configured vault.

## Chunk and embedding recipes

Version chunking, document embedding, and query embedding independently. The initial document recipe to evaluate is title + heading breadcrumb + selected aliases/tags + body. Keep authority metadata outside semantic text. Persist model name, FastEmbed package version, registry artifact source, actual dimension, all recipe versions, and corpus-policy version; incompatible changes require a controlled rebuild. Preserve/cache artifact checksums externally when byte-level provenance is required.

A configured model is valid only when its dimension and recipe are compatible with the active vector schema. Chunk-size, overlap, code/table/list handling, and long-paragraph behaviour are selected by evaluation rather than fixed by convention.

## Retrieval contract

The pipeline is:

1. Route query and select corpus profile.
2. Retrieve dense and BM25 candidates independently.
3. Preserve raw metrics and per-channel ranks in typed candidate records.
4. Fuse candidates with RRF.
5. Apply authority/filter policy and document diversification.
6. Optionally rerank when held-out evidence justifies cost.
7. Select bounded context and citations.

Vector distance, FTS5 BM25, RRF, and reranker scores remain distinct. RRF is not confidence. `rrf_k` and candidate limits are experiment defaults, not permanent product decisions.

## Evaluation

Before BM25, create at least 60 real-vault questions stratified across current state, decisions, exact terminology, concepts, evidence, chronology, ambiguity, and no-answer cases. Keep a held-out set. Record acceptable document/section sources, authority class, and expected retrieve/abstain mode.

Every evaluation run records git SHA, corpus fingerprint, model revision, recipe/config versions, Recall@K, MRR, nDCG, authority accuracy, duplicate concentration, citation fidelity, no-answer behaviour, p50/p95 latency, peak RSS, and failures. Compare vector-only, BM25-only, hybrid, and any reranker on the same held-out set. Optional stages ship only against predeclared quality and resource gates.

## Context and citations

Citation objects use vault-relative paths and include title, heading breadcrumb, stable chunk identity/hash, ordinal, source role, retrieval mode, excerpt boundaries, and truncation state. Context selection enforces total/per-document token budgets, source diversification, contiguous-only merging, conflict labels, and exact provenance.

All retrieved Markdown is untrusted reference data. Delimit and label it as evidence so embedded instructions cannot override host or user instructions.

## Hermes adapter

V1 tools:

- `llmwiki_search`: explicit profile-aware cited retrieval.
- `llmwiki_status`: freshness, integrity, model/recipe identity, and state.
- `llmwiki_reindex`: controlled, scope-explicit, status-visible maintenance.

The current Hermes `pre_llm_call` hook injects returned context into the current user message, receives sensitive conversation data, and is timeout-bounded/fail-open. V1.1 must use current-query-only input, persist no conversation history, remain visibly opt-in, and return no context on error or timeout. Plugin release requires `hermes plugins doctor --ci` and compatibility tests against a declared Hermes range.

## Privacy and operations

Document one-time model acquisition and an offline provisioning path; pin model revisions/checksums where possible. On POSIX, the projection directory is `0700` and the SQLite/WAL/SHM files are `0600`; unsupported platforms must report that they cannot verify an equivalent guarantee. Document backup/deletion behaviour because the projection contains source text, frontmatter, paths, and vectors. Do not log raw history or full retrieved content by default. Absolute paths are diagnostic-only. See [`operations.md`](operations.md) for current operational limits.

## Roadmap and release gates

### Stabilization

- Ordered migrations, true rebuild, integrity checks, corpus profiles, authority metadata, versioned recipes, and restrictive POSIX projection permissions are implemented. A real-vault vector baseline remains.
- Gate: zero derived-row orphans; fault-injection rollback passes; real-vault held-out evaluation and resource measurements recorded.

### V1 retrieval

- Add transactional FTS5/BM25, typed candidates, hybrid fusion, conditional reranking, context/citation objects, and explicit Hermes tools.
- Gate: every retrieval variant evaluated on the same held-out set; citation resolution structurally exact; authority and latency/resource targets met; plugin doctor and integration tests pass.

### V1.1 automatic retrieval

- Add deterministic routing, held-out confidence calibration, and opt-in `pre_llm_call` injection.
- Gate: predeclared injection precision/context-pollution target met; default remains off; timeout/error path verified fail-open.

### V2

- Add file watching, resolved wikilinks/backlinks, route-aware retrieval, temporal/project signals, and performance optimization only when each improves held-out results without authority regression.

### V3

Advanced GraphRAG, MCP, external APIs, additional agent integrations, and wiki write-back require demonstrated demand plus separate scope, privacy, and threat reviews.

## Non-goals before V1

Entity extraction, community detection, HyDE, query decomposition, remote embedding APIs, MCP, web UI, and source-writing automation are not V1 work.
