# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog; versions follow Semantic Versioning.

## [0.1.0] — 2026-09-01

First release candidate of the local-first, authority-aware wiki RAG and
its Hermes plugin.

### Added

- Incremental Obsidian indexer with heading-aware chunks, local FastEmbed
  embeddings (`BAAI/bge-small-en-v1.5`), sqlite-vec vectors, trigger-maintained
  FTS5, ordered transactional migrations (schema v8), integrity diagnostics,
  true full rebuild, restrictive projection permissions.
- Path-derived corpus metadata and profiles: `answer`, `project:<id>`,
  `evidence`, `history`, `all`.
- Hybrid retrieval (dense + BM25, reciprocal-rank fusion), intent-aware
  authority ordering with provenance conflict labels, document diversification,
  opt-in cross-encoder reranking.
- Citation objects and a budgeted context builder that wraps retrieved
  Markdown as untrusted reference material, with prompt-injection fixtures.
- Resolved wikilink graph, project-profile expansion, page-entity mention
  edges, deterministic communities, `llmwiki_related`.
- Hermes plugin (`llmwiki_search`, `llmwiki_status`, `llmwiki_reindex`,
  `llmwiki_related`), an inert-by-default calibrated `pre_llm_call` hook, and
  an opt-in in-host vault watcher.
- MCP stdio server exposing the same tools (`llmwiki mcp`).
- CLI: `init` (vault discovery, starter vault, persistent config, first index),
  `doctor`, `config show|set`, `index` (with `--watch`), `search`, `status`,
  `integrity`, `related`, `communities`, `bench`, `mcp`, and
  `eval validate|run|compare|regress|calibrate|report`.
- Persistent user config at `~/.config/llmwiki/config.toml`; the Hermes plugin
  falls back to it, and a `/llmwiki` slash command offers status, setup,
  reindex and doctor inside sessions.
- Evaluation harness with two real-vault golden sets, predeclared gates,
  reproducible run records, regression rule, and generated benchmarks.
- Documentation: architecture, install, configuration, tools, security,
  operations, evaluation, benchmarks.

### Measured and left off by default

- Cross-encoder reranking (Gate R: latency and memory), automatic injection
  (Gate A: coverage), linked-pages RRF channel, recency ordering, multi-query
  decomposition.

### Compatibility

- Python 3.11 – 3.14. Hermes Agent 0.20.6 (source 2026.8.27) tested; see
  `hermes_plugin/plugin.yaml`.
