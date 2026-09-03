# Changelog

All notable changes are recorded here. The format follows Keep a Changelog and
versions follow Semantic Versioning.

## [Unreleased]

## [0.1.0] - 2026-09-03

First public alpha release of llmwiki.

### Added

- Local incremental indexing for Markdown and Obsidian vaults with
  heading-aware chunks, local FastEmbed embeddings, sqlite-vec, and FTS5.
- Hybrid BM25 and vector retrieval with corpus profiles, authority-aware
  ordering, cited context, conflict labels, and document diversification.
- CLI setup, health checks, search, indexing, related-page lookup, configuration,
  evaluation, benchmarking, and diagnostics.
- MCP server and Hermes plugin exposing `llmwiki_search`, `llmwiki_status`,
  `llmwiki_reindex`, and `llmwiki_related`.
- Bundled `llmwiki:using-llmwiki` Hermes skill with retrieval-profile guidance,
  citation handling, related-page exploration, freshness checks, and safe
  reindex policy.
- Resolved wikilink graph, project-profile expansion, page mentions, and
  deterministic communities.
- Default-on, debounced Hermes watcher with read-only event filtering.
- Advisory update checks that never install software automatically.
- Adaptive embedding resource profiles, bounded batching, process-memory and
  available-memory diagnostics, and cgroup-v2 headroom reporting.
- Linux/macOS and Windows installers, persistent user configuration, offline
  provisioning guidance, security documentation, and a synthetic evaluation
  example.

### Changed

- Unchanged structural embedding inputs retain their chunk and vector IDs, so a
  small document edit embeds only the changed chunks.
- User documentation now separates installation and everyday use from
  contributor, architecture, evaluation, and release-engineering material.
- Linux/macOS and Windows installers now preserve checkout and installation
  paths containing spaces when invoking pip.

### Measured and disabled by default

- Cross-encoder reranking because it exceeded latency and memory gates.
- Automatic context injection because its held-out coverage gate did not pass.
- Linked-pages ranking, recency ordering, and multi-query decomposition because
  they did not improve the accepted evaluation baseline.

### Compatibility

- Python 3.11–3.14 on Linux and macOS.
- Native Windows for the CLI and MCP server; Hermes integration uses WSL.
- Hermes Agent 0.21.0 or newer; Plugin Doctor is tested against the declared
  lower bound and current Hermes main.
