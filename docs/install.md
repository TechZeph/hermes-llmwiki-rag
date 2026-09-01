# Install

Three ways to run llmwiki: standalone CLI, MCP server for any MCP-capable
agent, or Hermes plugin. All three share one projection database and the
same local models.

## Requirements

- Python 3.11 or newer (Hermes ships 3.11; the suite is run on 3.11 and 3.14).
- SQLite with FTS5 (standard on Linux and macOS Python builds).
- About 130 MB for the `BAAI/bge-small-en-v1.5` ONNX model, downloaded once
  by FastEmbed into `~/.cache/fastembed` (override with `FASTEMBED_CACHE_PATH`).
- Disk for the projection: roughly 3–4× the vault's Markdown size.

## Standalone CLI

```bash
git clone https://github.com/TechZeph/hermes-llmwiki-rag
cd hermes-llmwiki-rag
python -m venv .venv
.venv/bin/python -m pip install -e ".[mcp]"        # add [dev] for tests

export LLMWIKI_VAULT=~/Workspace/vaults/clanker-vault   # or pass --vault
.venv/bin/llmwiki index          # first run embeds every chunk (minutes; see below)
.venv/bin/llmwiki status
.venv/bin/llmwiki search --query "why did we choose sqlite-vec?"
```

The projection is created at `$XDG_DATA_HOME/llmwiki/llmwiki.sqlite`
(default `~/.local/share/llmwiki/llmwiki.sqlite`) with `0700`/`0600`
permissions. Set `LLMWIKI_DB` to move it. Never put it inside the vault.

First index: ~5,000 chunks take about 45 minutes on a 16-core CPU and hold
around 5.5 GB RSS while the ONNX arena is warm; later runs touch only
changed pages and finish in seconds.

Keep it fresh:

```bash
.venv/bin/llmwiki index --watch          # coalesced incremental reindex on change
```

or a user service:

```ini
# ~/.config/systemd/user/llmwiki-watch.service
[Unit]
Description=llmwiki vault watcher
[Service]
Environment=LLMWIKI_VAULT=%h/Workspace/vaults/clanker-vault
ExecStart=%h/Workspace/repos/hermes-llmwiki-rag/.venv/bin/llmwiki index --watch
Restart=on-failure
[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload && systemctl --user enable --now llmwiki-watch
```

## MCP server

```bash
.venv/bin/llmwiki mcp --vault ~/Workspace/vaults/clanker-vault [--watch]
```

Serves `llmwiki_search`, `llmwiki_status`, `llmwiki_reindex` and
`llmwiki_related` over stdio. Example client configuration (Claude Code,
Codex, and other MCP hosts use the same shape):

```json
{
  "mcpServers": {
    "llmwiki": {
      "command": "/home/you/hermes-llmwiki-rag/.venv/bin/llmwiki",
      "args": ["mcp", "--vault", "/home/you/vault"]
    }
  }
}
```

Run `llmwiki index` once before starting the server; `--watch` refuses a
cold start on purpose.

## Hermes plugin

```bash
~/.hermes/hermes-agent/venv/bin/pip install -e /path/to/hermes-llmwiki-rag
ln -s /path/to/hermes-llmwiki-rag/hermes_plugin ~/.hermes/plugins/llmwiki
hermes plugins enable llmwiki --no-allow-tool-override
hermes config set plugins.entries.llmwiki.settings.vault /path/to/vault
hermes plugins doctor /path/to/hermes-llmwiki-rag/hermes_plugin --ci
hermes gateway restart          # running sessions load plugins at start
```

Or from GitHub once published: `hermes plugins install TechZeph/hermes-llmwiki-rag/hermes_plugin`.

Settings reference: `docs/configuration.md`. Tool contract: `docs/tools.md`.

## Uninstall

```bash
hermes plugins disable llmwiki && rm ~/.hermes/plugins/llmwiki
~/.hermes/hermes-agent/venv/bin/pip uninstall hermes-llmwiki-rag
rm -f ~/.local/share/llmwiki/llmwiki.sqlite{,-wal,-shm}      # projection
rm -rf ~/.cache/fastembed                                     # models (optional)
```

The vault is never modified by any of the above.
