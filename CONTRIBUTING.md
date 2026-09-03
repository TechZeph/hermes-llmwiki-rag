# Contributing to llmwiki

Thanks for improving llmwiki. User documentation and development documentation live on the same `main` branch so changes to behavior and instructions can be reviewed together. A separate development branch would let them drift and is not used.

## Ways to contribute

- Report a bug or propose a feature through [GitHub Issues](https://github.com/TechZeph/llmwiki-rag/issues).
- Improve installation, platform support, retrieval quality, tests, or documentation.
- Fork the repository to build a custom retriever, host adapter, embedding backend, or user interface.
- Report security problems privately using the process in
  [SECURITY.md](SECURITY.md), not a public issue.

## Development setup

Requirements: Python 3.11 or newer and SQLite with FTS5.

```bash
git clone https://github.com/TechZeph/llmwiki-rag.git
cd llmwiki-rag
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

On Windows, replace `.venv/bin/` with `.venv\\Scripts\\`.

## Required checks

Run these before opening a pull request:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy llmwiki hermes_plugin
bash -n install.sh
```

PowerShell installer changes are also exercised by the Windows CI job. Changes to the Hermes adapter must pass Plugin Doctor:

```bash
hermes plugins doctor hermes_plugin --ci
```

## Evaluate retrieval changes

Do not use a personal vault in public test fixtures. The repository includes a synthetic vault and golden set:

```bash
sample_dir="$(mktemp -d)"
.venv/bin/llmwiki index \
  --vault evals/sample-vault \
  --db "$sample_dir/index.sqlite"
.venv/bin/llmwiki eval run \
  --set evals/golden/sample-vault.json \
  --vault evals/sample-vault \
  --db "$sample_dir/index.sqlite" \
  --split all \
  --out "$sample_dir/runs"
```

Changes to chunking, embeddings, ranking, authority policy, or context selection should include a regression comparison under the protocol in [docs/evaluation.md](docs/evaluation.md). Do not hand-edit [docs/benchmarks.md](docs/benchmarks.md); regenerate it with `llmwiki eval report` from appropriate run records.

## Pull requests

1. Fork the repository and create a focused branch from `main`.
2. Add or update tests for behavior changes.
3. Update the user documentation when behavior or configuration changes.
4. Run the required checks above.
5. Open a pull request explaining the problem, the change, and the verification performed.

Keep pull requests focused. Avoid unrelated formatting, renames, or refactors.

## Architecture and extension points

Read [docs/architecture.md](docs/architecture.md) before changing storage, indexing, retrieval, citations, or host integration. The stable boundary is:

- `llmwiki/` contains the host-independent indexing and retrieval engine.
- `hermes_plugin/` adapts that engine to Hermes.
- `llmwiki mcp` exposes the same service to MCP hosts.
- The selected Markdown vault remains canonical and read-only; SQLite is a disposable projection.

Forks can add integrations without modifying the core by depending on `llmwiki.service.WikiService` and preserving the security contract documented in [docs/security.md](docs/security.md).
