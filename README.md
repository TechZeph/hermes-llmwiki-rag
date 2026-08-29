# hermes-llmwiki-rag

Local-first hybrid retrieval-augmented generation over an Obsidian vault, packaged as a standalone plugin for [Nous Hermes Agent](https://nousresearch.com).

## What it does

- Treats an Obsidian vault as the canonical source of truth.
- Indexes Markdown notes incrementally (new / modified / deleted / unchanged).
- Will combine dense vector search, FTS5/BM25 lexical search, reciprocal-rank fusion, cross-encoder reranking, and Obsidian graph traversal. (Phases 3+.)
- Will inject high-confidence retrieval into Hermes through a `pre_llm_call` hook. (Phase 9.)
- Runs locally. SQLite + FTS5 + sqlite-vec. Local embeddings via FastEmbed. Local reranker.

## Status

**Phase 1 in progress** (project setup + Obsidian indexer). See `wiki/projects/hermes-llmwiki-rag/plan.md` for the full 16-phase plan.

The retrieval engine, the Hermes plugin, and the automatic context injection are not yet implemented. The package currently provides the project skeleton, configuration, logging, the indexer interface, and a working CLI that can index an Obsidian vault into a local SQLite database.

## Quick start

```bash
# 1. install (editable, with dev tools)
uv venv --python 3.14 .venv
.venv/bin/python -m pip install -e ".[dev]"

# 2. index a vault
.venv/bin/llmwiki index \
  --vault ~/Workspace/vaults/clanker-vault \
  --db   ./.data/llmwiki.sqlite

# 3. inspect the database
.venv/bin/llmwiki status --db ./.data/llmwiki.sqlite
```

## Architecture

The RAG core is independent of Hermes. The Hermes plugin is a thin adapter. Either can be replaced or extended without rewriting the other.

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
  citations.py    Phase 7: source attribution
  context.py      Phase 7: LLM-ready context blocks
  cli.py          click CLI
  db.py           SQLite schema + connection helpers

hermes_plugin/    Phase 8+: Hermes plugin adapter
tests/
  unit/           isolated unit tests
  integration/    filesystem + SQLite tests
  eval/           Phase 13+: golden-question evaluation harness
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
