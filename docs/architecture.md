# Architecture

> Contributor reference. Start with the [user documentation](README.md) if you
> only want to install or use llmwiki.

llmwiki is a host-independent indexing and retrieval engine with thin adapters
for the command line, MCP, and Hermes Agent.

## System boundaries

- The selected Markdown vault is canonical and read-only.
- SQLite, FTS5, sqlite-vec, graph edges, and embeddings form a disposable local
  projection outside the vault.
- The retrieval core does not import Hermes.
- The Hermes plugin and MCP server use the same `WikiService` interface.
- Retrieved Markdown is untrusted evidence, never executable instructions.

## Indexing pipeline

1. Discover regular Markdown files contained within the configured vault.
2. Parse frontmatter, headings, tags, aliases, and wikilinks.
3. Create heading-aware structural chunks.
4. Classify source kind, page role, project, and route-map status from the path.
5. Generate local FastEmbed vectors and maintain FTS5 rows.
6. Resolve wikilinks, mentions, and deterministic link communities.
7. Commit each document's metadata, chunks, vectors, and search rows
   transactionally.

Incremental indexing hashes files and embedding inputs. Unchanged chunks keep
their identifiers and vectors; new, changed, and deleted content updates only
the affected projection rows. Ordered migrations and integrity checks protect
existing projections.

## Retrieval pipeline

1. Validate the corpus profile and optional date filter.
2. Retrieve dense and BM25 candidates independently.
3. Preserve channel-specific metrics and ranks.
4. Fuse candidates with reciprocal-rank fusion.
5. Apply intent-aware authority ordering and document diversification.
6. Optionally apply experimental channels when explicitly enabled.
7. Build bounded excerpts, citations, conflict labels, and an untrusted-context
   envelope.

Vector distance, FTS5 BM25, fusion scores, and optional reranker scores remain
separate. A fusion score is a ranking signal, not confidence.

## Corpus profiles and authority

- `answer`: curated wiki pages, excluding history and route maps.
- `project:<id>`: one project workspace plus linked curated pages.
- `evidence`: raw sources and clippings.
- `history`: root and project logs.
- `all`: the complete corpus for diagnostics.

Authority depends on query intent. Current-state pages lead status questions;
decision pages lead rationale questions; logs lead chronology questions; and
raw sources support evidence requests. Conflicts are labelled with provenance
instead of silently resolved by score.

## Context and citations

Every result carries a vault-relative path, title, heading breadcrumb, chunk
ordinal and hash, authority class, source kind, and channel ranks. Context
selection enforces total and per-document budgets, document diversification,
contiguous-only merging, and explicit truncation.

Context is wrapped in fixed delimiters and neutralised so text inside a vault
cannot forge the envelope. Host applications must continue to treat retrieved
content as data.

## Host integrations

`llmwiki.service.WikiService` is the shared public boundary. It powers:

- CLI search, status, related-page lookup, and indexing;
- four MCP tools over stdio;
- the Hermes plugin's four tools, slash command, watcher, and optional
  `pre_llm_call` hook.

Automatic injection is registered but disabled by default. When enabled, it
uses only the current query, a deterministic router, a calibrated gate, and a
bounded deadline. Errors, timeouts, and rejected gates inject nothing.

## Storage and privacy

The projection contains source text, metadata, paths, hashes, and embeddings.
On POSIX, llmwiki creates its directory with mode `0700` and database/WAL/SHM
files with mode `0600`. Other platforms receive a warning rather than a false
permission guarantee.

The package has no telemetry client and does not persist query text or
conversation history. Model acquisition is the only expected network operation
during normal setup; advisory update checks can be disabled.

## Resource model

Embedding work uses one bounded batch controller across initial indexing and
changed-document updates. The `conservative`, `balanced`, and `performance`
profiles select default batch and thread counts; explicit limits can override
them. Process RSS, available memory, and Linux cgroup-v2 headroom are advisory
signals. Hard limits remain the responsibility of the host operating system.

## Evaluation contract

Changes to chunking, embeddings, ranking, authority policy, or context selection
must be compared on the same held-out corpus fingerprint. Optional stages ship
enabled only when they pass predeclared quality, authority, latency, and memory
gates. See [Evaluation](evaluation.md) and [Benchmarks](benchmarks.md).

## Extension points

- Add a host adapter around `WikiService` without importing host code into the
  core.
- Add a retrieval experiment behind an explicit disabled-by-default setting and
  the regression protocol.
- Add a model backend only with dimension, recipe, provenance, offline, and
  resource behavior defined.
- Add a schema migration as the next ordered, transactional version with
  rollback and integrity coverage.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for setup and required checks.