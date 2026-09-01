# hermes-llmwiki-rag

Local-first, authority-aware retrieval over an Obsidian LLM wiki, packaged as a
standalone plugin for [Nous Hermes Agent](https://hermes-agent.nousresearch.com/docs/).
Markdown in the vault is canonical; SQLite (FTS5 + sqlite-vec) is a rebuildable
projection. Retrieval is hybrid (BM25 + dense with reciprocal-rank fusion),
scoped by explicit corpus profiles, ordered by a deterministic authority policy,
and returned as cited, budgeted, untrusted-reference context.

## Status

**V1, V2 and V3 delivered; V1.1 calibrated but opt-in.** See
`docs/evaluation.md` for the predeclared gates and every recorded decision,
and `docs/benchmarks.md` for the generated results tables.

| Stage | State |
|---|---|
| 0 Stabilization (migrations, integrity, profiles, recipes, privacy, eval baseline) | done |
| 1 FTS5/BM25 + hybrid RRF (Gate H passed on held-out; hybrid is the default) | done |
| 2 Citations, budgeted context, injection boundaries; reranker measured (Gate R failed, opt-in) | done |
| 3 Hermes plugin: `llmwiki_search`, `llmwiki_status`, `llmwiki_reindex`; `hermes plugins doctor --ci` passes | done |
| 4 Routing + calibrated injection gate (safety clauses pass, coverage clause fails; opt-in only) | done |
| V2: resolved wikilink graph, project-profile expansion, in-plugin watcher, date filters; linked-pages channel and recency measured and left off | done |
| V3: MCP server, page-entity mention graph + communities, `llmwiki_related`, multi-query decomposition (measured, off), production docs, generated benchmarks | done |
| Stage 5: Hermes ecosystem release (PyPI, CI matrix, index listing) | planned, see the vault plan |

Held-out numbers with the shipped configuration (equal-weight RRF, k=20):

| golden set | variant | hit@5 | recall@10 | MRR | nDCG@10 | authority@1 | p95 |
|---|---|---|---|---|---|---|---|
| v1.1 (37 q, vault wording) | hybrid | 0.970 | 0.907 | 0.860 | 0.835 | 0.879 | 112 ms |
| v2 (26 q, paraphrased) | hybrid | 0.957 | 0.822 | 0.857 | 0.777 | 0.870 | 115 ms |

Single channels for comparison on v2 held-out: lexical hit@5 0.870 / MRR
0.833, dense 0.783 / 0.707. Full tables per set, split and category:
`docs/benchmarks.md`.

## What it does

- Indexes a vault incrementally (new / modified / deleted / unchanged) with
  heading-aware chunks, local FastEmbed embeddings (`BAAI/bge-small-en-v1.5`),
  sqlite-vec vectors, and a trigger-maintained FTS5 index — all replaced
  atomically per document.
- Classifies every page deterministically from its path (`source_kind`,
  `page_role`, `project_id`, route-map flag) and exposes corpus profiles:
  `answer` (curated wiki, default), `project:<id>`, `evidence` (raw sources and
  clippings), `history` (append-only logs), `all` (diagnostics).
- Retrieves dense and BM25 candidates independently, fuses with weighted RRF,
  applies an intent-aware authority re-ordering (current-state pages for status
  questions, decisions for rationale, logs for chronology), diversifies by
  document, and labels provenance conflicts instead of resolving them.
- Builds citation objects (vault-relative path, breadcrumb, chunk ordinals,
  content hashes) and a context block with total/per-document token budgets,
  contiguous-only merging, and fixed delimiters that mark retrieved Markdown as
  untrusted evidence. Prompt-injection fixtures are part of the test suite.
- Resolves Obsidian wikilinks into a graph (99% resolved on the reference
  vault), expands `project:<id>` with linked curated pages, adds page-entity
  mention edges and deterministic link communities, and answers "what is
  related to this page" through `llmwiki_related`.
- Ships a Hermes plugin with four explicit tools and an inert-by-default
  `pre_llm_call` hook that only injects when the operator opts in, the
  deterministic router says retrieve, the calibrated gate passes, and the
  internal deadline is met. It never persists conversation history or query text.
- Records every evaluation run with git SHA, corpus fingerprint, recipe/model
  identity, IR/authority/citation metrics, latency and peak RSS.

## Quick start

```bash
# 1. install (Python 3.11+; editable, with dev tools)
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"

# 2. index a vault (first run embeds every chunk: ~45 min for 5k chunks)
.venv/bin/llmwiki index --vault ~/Workspace/vaults/clanker-vault

# 3. search (hybrid by default) and print the LLM-ready context block
.venv/bin/llmwiki search --query "why did we choose sqlite-vec?" --profile answer
.venv/bin/llmwiki search --query "when did FTS5 land?" --profile history --context

# 4. inspect
.venv/bin/llmwiki status
.venv/bin/llmwiki integrity --vault ~/Workspace/vaults/clanker-vault
```

The projection lives at `$XDG_DATA_HOME/llmwiki/llmwiki.sqlite` (default
`~/.local/share/llmwiki/llmwiki.sqlite`) with `0700`/`0600` permissions. Set
`LLMWIKI_VAULT` / `LLMWIKI_DB` to avoid repeating flags.

## MCP server (any agent)

```bash
.venv/bin/llmwiki mcp --vault ~/Workspace/vaults/clanker-vault --watch
```

Serves the same four tools over stdio for Claude Code, Codex, or any MCP host.
Client config and details: `docs/install.md`.

## Hermes plugin

```bash
# install the core into Hermes' Python and link the plugin directory
~/.hermes/hermes-agent/venv/bin/pip install -e /path/to/hermes-llmwiki-rag
ln -s /path/to/hermes-llmwiki-rag/hermes_plugin ~/.hermes/plugins/llmwiki
hermes plugins enable llmwiki --no-allow-tool-override
hermes config set plugins.entries.llmwiki.settings.vault /path/to/vault
hermes plugins doctor /path/to/hermes-llmwiki-rag/hermes_plugin --ci
```

Settings (all under `plugins.entries.llmwiki.settings`) are listed in
`docs/configuration.md`; the tool contract is in `docs/tools.md`. Set
`watch: true` to keep the projection fresh from inside the gateway (it refuses
a cold start; run `llmwiki index` once first). Full rebuilds need
`allow_full_rebuild: true` and `confirm: true` on the call. Automatic injection
is off by default and requires the shipped gate to be safety-certified.

## Evaluation

```bash
.venv/bin/llmwiki eval validate --set private/evals/golden/clanker-vault-v1.json --vault ~/Workspace/vaults/clanker-vault
.venv/bin/llmwiki eval run --set private/evals/golden/clanker-vault-v1.json --vault ~/Workspace/vaults/clanker-vault \
    --variant dense --variant lexical --variant hybrid --split heldout
.venv/bin/llmwiki eval compare evals/runs/*.json
.venv/bin/llmwiki eval calibrate --set private/evals/golden/clanker-vault-v1.json --vault ~/Workspace/vaults/clanker-vault
```

Tune on `dev`, report `heldout`. Gates and recorded decisions: `docs/evaluation.md`.
The real-vault golden sets and run records are private (see below); the public
repo ships `evals/sample-vault/` with `evals/golden/sample-vault.json` so the
harness runs end to end for anyone:

```bash
.venv/bin/llmwiki index --vault evals/sample-vault --db /tmp/sample.sqlite
.venv/bin/llmwiki eval run --set evals/golden/sample-vault.json --vault evals/sample-vault --db /tmp/sample.sqlite --split all
```

## Public and private parts of the repository

Public (this repository): the `llmwiki` package, `hermes_plugin`, tests, docs,
CI, the golden-set schema, the synthetic sample vault and set, generated
benchmark tables, the shipped injection gate, and the release scripts.

Private (`private/`, git-ignored, never published): the real-vault golden
question sets and their drafts, recorded evaluation runs (they embed queries,
page paths and section titles from a personal vault), and release scratch such
as announcement drafts and upstream patches in flight. `evals/runs/` is
git-ignored too so nobody commits run records by accident.

## Architecture

```
llmwiki/
  config.py       immutable Settings (env for CLI bootstrap only)
  db.py           ordered transactional migrations (v8), integrity, connection
  parser.py       frontmatter, headings, tags, aliases, raw wikilinks
  chunker.py      heading-aware structural chunking
  corpus.py       path-derived corpus metadata + profile predicates
  recipes.py      versioned chunker/document/query recipes
  embeddings.py   FastEmbed embedder + provenance
  vector.py       sqlite-vec store
  lexical.py      FTS5 BM25 index, safe query construction
  hybrid.py       weighted RRF + document diversification
  authority.py    intent detection, authority ordering, conflict labels
  reranker.py     FastEmbed cross-encoder (opt-in)
  retrieval.py    Retriever: profiles → channels → fusion → policy → candidates
  citations.py    Citation objects, budgeted untrusted-evidence context
  routing.py      deterministic retrieve/profile routing
  confidence.py   calibrated injection gate
  graph.py        resolved wikilinks, neighbours, project-profile expansion
  entities.py     page-entity mentions, communities, related pages
  multiquery.py   deterministic decomposition + fusion (experiment)
  watch.py        coalescing file watcher and the supervised in-host watcher
  service.py      WikiService: the one engine behind CLI, MCP and Hermes
  mcp_server.py   MCP stdio server over WikiService
  indexer.py      incremental, atomic vault → projection
  evaluation/     golden sets, metrics, run records, calibration, reports
  cli.py          click CLI
hermes_plugin/    plugin.yaml, register(ctx), tools, injection_gate.json (thin adapter)
tests/            unit + integration (synthetic vaults, fake embedders, injection fixtures)
evals/            golden question sets and recorded runs
docs/             architecture, operations, evaluation
```

Docs: `architecture.md` (contract), `install.md`, `configuration.md`,
`tools.md`, `security.md`, `operations.md`, `evaluation.md` (gates and
decisions), `benchmarks.md` (generated).

## Development

```bash
.venv/bin/pytest -q
.venv/bin/ruff check . && .venv/bin/ruff format .
.venv/bin/mypy llmwiki hermes_plugin
```

## License

MIT. See `LICENSE`.
