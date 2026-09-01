# hermes-llmwiki-rag

Local-first retrieval over an Obsidian vault, designed as a standalone plugin for [Nous Hermes Agent](https://hermes-agent.nousresearch.com/docs/). The target is an authority-aware hybrid pipeline; the implementation at commit `44c6e56` is a vector-only semantic index.

## What it does

- Treats Markdown as canonical and SQLite/FTS/vector data as a rebuildable projection.
- Indexes Markdown notes incrementally (new / modified / deleted / unchanged).
- Currently provides heading-aware chunks, local FastEmbed embeddings, sqlite-vec storage, and vector-only CLI search.
- Will add explicit corpus profiles and intent-specific authority policy before BM25/hybrid fusion.
- Will add FTS5/BM25 and RRF; reranking ships only if held-out evaluation justifies its latency and memory.
- Will expose explicit, cited Hermes tools for V1. Optional `pre_llm_call` injection is a calibrated, opt-in V1.1 feature.
- Runs locally after one-time model provisioning. Retrieved Markdown is treated as untrusted reference data.

## Status

**Stage 0 stabilization after Phase 3 semantic retrieval.** Incremental indexing, structural chunking, local embeddings, sqlite-vec persistence, and vector-only CLI search are implemented and tested.

Before FTS5, the project is defining corpus and authority policy, versioning embedding recipes, and establishing a real-vault held-out baseline. Ordered migrations, a true `index --mode full` rebuild, a read-only `integrity` command, transactional document projection updates, and source-deletion vector cleanup are implemented. BM25, integrated hybrid retrieval, reranking, authority-aware context/citations, Hermes tools, routing, and automatic injection are not implemented yet.

See [`docs/architecture.md`](docs/architecture.md) for the current architecture, measurable release gates, Hermes hook constraints, privacy requirements, and scope boundaries.

## Quick start

```bash
# 1. install (editable, with dev tools)
uv venv --python 3.14 .venv
.venv/bin/python -m pip install -e ".[dev]"

# 2. index a vault
.venv/bin/llmwiki index \
  --vault ~/Workspace/vaults/clanker-vault \
  --db   ./.data/llmwiki.sqlite

# 3. search the semantic index
.venv/bin/llmwiki search \
  --db ./.data/llmwiki.sqlite \
  --query "how does retrieval authority work?"

# 4. inspect the database
.venv/bin/llmwiki status --db ./.data/llmwiki.sqlite
```

## Architecture

The retrieval core is independent of Hermes. The Hermes plugin is a thin adapter. Markdown remains canonical; the database can be rebuilt. See [`docs/architecture.md`](docs/architecture.md) for corpus profiles, authority rules, integrity requirements, evaluation, and release boundaries.

```
llmwiki/
  config.py       configuration
  logging.py      structured logging
  models.py       dataclasses (Document, Chunk, ...)
  parser.py       markdown + frontmatter parsing
  chunker.py      structural chunking
  indexer.py      incremental vault → SQLite
  embeddings.py   Phase 3: FastEmbed wrapper
  vector.py       Phase 3: sqlite-vec wrapper
  lexical.py      Phase 4: FTS5/BM25
  hybrid.py       Phase 5: reciprocal-rank fusion
  reranker.py     Phase 6: cross-encoder reranker
  graph.py        Phase 10: Obsidian wikilink graph
  retrieval.py    Phase 5+: end-to-end retrieval
  scoring.py      Phase 6+: score normalisation + combination
  citations.py    Phase 7: source attribution + placeholder context builder
  cli.py          click CLI
  db.py           SQLite schema + connection helpers

hermes_plugin/    Phase 8+: Hermes plugin adapter
tests/
  unit/           isolated unit tests
  integration/    filesystem + SQLite tests
  eval/           Stage 0+: real-vault golden-question evaluation harness
```

## Development

```bash
# tests
.venv/bin/pytest -q

# lint + format
.venv/bin/ruff check .
.venv/bin/ruff format .

# type check
.venv/bin/mypy llmwiki hermes_plugin
```

## License

MIT. See `LICENSE`.
