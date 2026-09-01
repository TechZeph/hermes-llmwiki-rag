# hermes-llmwiki-rag

Local-first, authority-aware retrieval over an Obsidian LLM wiki, packaged as a
standalone plugin for [Nous Hermes Agent](https://hermes-agent.nousresearch.com/docs/).
Markdown in the vault is canonical; SQLite (FTS5 + sqlite-vec) is a rebuildable
projection. Retrieval is hybrid (BM25 + dense with reciprocal-rank fusion),
scoped by explicit corpus profiles, ordered by a deterministic authority policy,
and returned as cited, budgeted, untrusted-reference context.

## Status

**V1 complete, V1.1 calibrated but opt-in.** See `docs/evaluation.md` for the
predeclared gates and the recorded held-out results.

| Stage | State |
|---|---|
| 0 Stabilization (migrations, integrity, profiles, recipes, privacy, eval baseline) | done |
| 1 FTS5/BM25 + hybrid RRF (Gate H passed on held-out; hybrid is the default) | done |
| 2 Citations, budgeted context, injection boundaries; reranker measured (Gate R failed, opt-in) | done |
| 3 Hermes plugin: `llmwiki_search`, `llmwiki_status`, `llmwiki_reindex`; `hermes plugins doctor --ci` passes | done |
| 4 Routing + calibrated injection gate (safety clauses pass, coverage clause fails; opt-in only) | done |
| 5 V2: wikilink graph, file watching, temporal signals | see `docs/architecture.md` |

Held-out numbers (37 questions, `evals/golden/clanker-vault-v1.json`):

| variant | hit@5 | recall@10 | MRR | nDCG@10 | authority@1 | p95 |
|---|---|---|---|---|---|---|
| dense | 0.848 | 0.786 | 0.773 | 0.726 | 0.848 | 77 ms |
| lexical | 0.939 | 0.864 | 0.831 | 0.815 | 0.879 | 6 ms |
| **hybrid (default)** | **0.939** | **0.889** | **0.836** | **0.818** | **0.879** | 83 ms |

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
- Ships a Hermes plugin with three explicit tools and an inert-by-default
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

## Hermes plugin

```bash
# install the core into Hermes' Python and link the plugin directory
~/.hermes/hermes-agent/venv/bin/pip install -e /path/to/hermes-llmwiki-rag
ln -s /path/to/hermes-llmwiki-rag/hermes_plugin ~/.hermes/plugins/llmwiki
hermes plugins enable llmwiki --no-allow-tool-override
hermes config set plugins.entries.llmwiki.settings.vault /path/to/vault
hermes plugins doctor /path/to/hermes-llmwiki-rag/hermes_plugin --ci
```

Settings (all under `plugins.entries.llmwiki.settings`): `vault` (required),
`db`, `default_profile`, `retrieval_mode`, `max_results`,
`context_budget_tokens`, `rerank`, `allow_reindex`, `allow_full_rebuild`,
`stale_after_hours`, `auto_inject`, `auto_inject_profile`,
`auto_inject_deadline_ms`, `auto_inject_budget_tokens`. Full rebuilds need
`allow_full_rebuild: true` and `confirm: true` on the call. Automatic injection
is off by default and requires the shipped gate to be safety-certified.

## Evaluation

```bash
.venv/bin/llmwiki eval validate --set evals/golden/clanker-vault-v1.json --vault ~/Workspace/vaults/clanker-vault
.venv/bin/llmwiki eval run --set evals/golden/clanker-vault-v1.json --vault ~/Workspace/vaults/clanker-vault \
    --variant dense --variant lexical --variant hybrid --split heldout
.venv/bin/llmwiki eval compare evals/runs/*.json
.venv/bin/llmwiki eval calibrate --set evals/golden/clanker-vault-v1.json --vault ~/Workspace/vaults/clanker-vault
```

Tune on `dev`, report `heldout`. Gates and recorded decisions: `docs/evaluation.md`.

## Architecture

```
llmwiki/
  config.py       immutable Settings (env for CLI bootstrap only)
  db.py           ordered transactional migrations (v6), integrity, connection
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
  indexer.py      incremental, atomic vault → projection
  evaluation/     golden sets, metrics, run records, calibration
  cli.py          click CLI
hermes_plugin/    plugin.yaml, register(ctx), runtime, tools, injection_gate.json
tests/            unit + integration (synthetic vaults, fake embedders, injection fixtures)
evals/            golden question sets and recorded runs
docs/             architecture, operations, evaluation
```

See `docs/architecture.md` for the product and retrieval contract and
`docs/operations.md` for local-data, offline provisioning, and backup limits.

## Development

```bash
.venv/bin/pytest -q
.venv/bin/ruff check . && .venv/bin/ruff format .
.venv/bin/mypy llmwiki hermes_plugin
```

## License

MIT. See `LICENSE`.
