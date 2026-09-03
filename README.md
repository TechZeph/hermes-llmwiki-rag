# llmwiki

Local, cited search for an Obsidian vault, available as a command-line tool,
an MCP server, and a plugin for
[Nous Hermes Agent](https://hermes-agent.nousresearch.com/docs/).

Your Markdown remains the source of truth. llmwiki reads it, builds a private
SQLite search index outside the vault, and combines keyword and semantic search
to find the most useful sections. Results include vault-relative citations so
you can open and verify the source.

## Why use it?

- **Local-first:** indexing and search run on your machine after the embedding
  model is downloaded once.
- **Your vault stays untouched:** llmwiki never writes, moves, or deletes vault
  files.
- **Better than one search method:** BM25 keyword search and local vector search
  are combined by default.
- **Useful results, not loose snippets:** headings, page roles, project scope,
  wikilinks, and authority are preserved.
- **Works with multiple agents:** use the CLI directly, connect any MCP client,
  or add the native Hermes tools.
- **No telemetry:** queries, retrieved text, and conversation history are not
  sent to the project or persisted by llmwiki.

## Status

Version 0.1.0 is the first public alpha release. The indexing, search, watcher,
CLI, MCP, and Hermes-plugin paths are implemented and verified. Automatic
context injection and the optional reranker remain opt-in while their release
gates continue to be evaluated.

## Install

Requirements: Python 3.11 or newer and SQLite with FTS5.

### CLI and MCP

```bash
git clone https://github.com/TechZeph/hermes-llmwiki-rag.git
cd hermes-llmwiki-rag
./install.sh
```

The installer creates an isolated environment under
`~/.local/share/llmwiki/venv`, adds a launcher at
`~/.local/bin/llmwiki`, and starts interactive setup.

### Hermes plugin

```bash
git clone https://github.com/TechZeph/hermes-llmwiki-rag.git
cd hermes-llmwiki-rag
./install.sh --hermes
hermes gateway restart
```

The plugin uses the vault selected during setup and keeps its index current in
the background by default. Set `watch: false` in the plugin settings to opt out.

Windows CLI and MCP installation, manual setup, upgrades, and uninstall steps
are in the [installation guide](docs/install.md).

## First use

If setup did not already run:

```bash
llmwiki init
```

`init` finds an Obsidian vault or creates a starter vault, saves the selection,
and builds the first index. The initial run downloads a local embedding model
and can take a while for a large vault; later updates index only changed pages.

Check the installation and search:

```bash
llmwiki doctor
llmwiki search --query "what decisions did we make about backups?"
llmwiki search --query "what changed this week?" --profile history --context
```

The default `answer` profile searches curated wiki pages. Other profiles target
one project (`project:<id>`), raw evidence (`evidence`), chronological logs
(`history`), or the whole vault for diagnostics (`all`).

## Connect an MCP client

Run the server over stdio:

```bash
llmwiki mcp --watch
```

It exposes four tools:

- `llmwiki_search` — cited hybrid retrieval.
- `llmwiki_status` — health, freshness, and index integrity.
- `llmwiki_reindex` — controlled incremental or confirmed full indexing.
- `llmwiki_related` — pages connected through links, mentions, or communities.

Client configuration examples are in the
[installation guide](docs/install.md#mcp-server).

## Documentation

### Using llmwiki

- [Installation](docs/install.md)
- [Configuration](docs/configuration.md)
- [Commands and tools](docs/tools.md)
- [Security and privacy](docs/security.md)
- [Security reporting](SECURITY.md)
- [Operations and offline use](docs/operations.md)

### Contributing or building on llmwiki

Development material is kept separate from the user guide:

- [Contributing and development setup](CONTRIBUTING.md)
- [Architecture](docs/architecture.md)
- [Evaluation protocol](docs/evaluation.md)
- [Benchmarks](docs/benchmarks.md)

See the [documentation index](docs/README.md) for the complete map. Development
documentation stays on `main` with the code so behavior and instructions cannot
drift; contributors work in normal fork or feature branches.

## Privacy summary

The selected vault is read-only. The generated SQLite projection contains vault
text and embeddings, so llmwiki stores it outside the vault with restrictive
permissions on POSIX systems. It does not persist query history or conversation
history and has no telemetry client. See [Security and privacy](docs/security.md)
before enabling automatic context injection.

## License

MIT. See [LICENSE](LICENSE).
