# Install llmwiki

llmwiki can run as a standalone command-line tool, an MCP server, or a Hermes
plugin. All three use the same local index and embedding model.

## Requirements

- Python 3.11 or newer.
- SQLite with FTS5, included with standard Python builds on Linux, macOS, and
  Windows from python.org.
- About 130 MB for the local `BAAI/bge-small-en-v1.5` model, downloaded once.
- Disk space for the generated index, typically three to four times the size of
  the vault's Markdown files.

Obsidian is optional. llmwiki works with any folder of Markdown files, although
it understands Obsidian wikilinks and vault structure.

## Linux and macOS

```bash
git clone https://github.com/TechZeph/hermes-llmwiki-rag.git
cd hermes-llmwiki-rag
./install.sh
```

The installer:

1. checks Python and SQLite FTS5;
2. creates an isolated environment at `~/.local/share/llmwiki/venv`;
3. adds `~/.local/bin/llmwiki`;
4. runs `llmwiki init` to select a vault and build the first index.

If `~/.local/bin` is not on `PATH`, the installer identifies the directory you
need to add. Check the completed installation with:

```bash
llmwiki doctor
```

Useful installer options:

```text
./install.sh --hermes             also install and enable the Hermes plugin
./install.sh --no-init            install without running setup
./install.sh --dry-run            show what would change
./install.sh -- /path/to/vault    pass a vault directly to setup
```

## Windows

Native Windows supports the CLI and MCP server. Hermes uses the Linux installer
inside WSL.

```powershell
git clone https://github.com/TechZeph/hermes-llmwiki-rag.git
cd hermes-llmwiki-rag
.\install.ps1
```

Configuration is stored in `%APPDATA%\llmwiki\config.toml`; the generated index
is stored in `%LOCALAPPDATA%\llmwiki\llmwiki.sqlite`. Windows permissions are not
changed automatically, so keep that directory restricted to your user account.

## First setup

The installer normally starts setup for you. To run it again:

```bash
llmwiki init
llmwiki init /path/to/vault
llmwiki init --create ~/llmwiki-vault
```

`init` saves the selected vault, creates the generated index outside it, and
runs the first index. The first run downloads the embedding model and may take
time on a large vault. Later runs process only new, changed, or deleted pages.

The default generated index is
`$XDG_DATA_HOME/llmwiki/llmwiki.sqlite`, or
`~/.local/share/llmwiki/llmwiki.sqlite` when `XDG_DATA_HOME` is unset. Never put
the generated index inside the vault.

## Standalone CLI

After setup:

```bash
llmwiki status
llmwiki search --query "what changed this week?" --profile history
llmwiki related wiki/projects/example/current-state.md
```

To keep a standalone process watching for vault changes:

```bash
llmwiki index --watch
```

Run only one watcher for a given index. If the Hermes plugin or an MCP server is
already watching, do not start a second standalone watcher.

## MCP server

Run the stdio server after completing the first index:

```bash
llmwiki mcp --watch
```

Example MCP client configuration:

```json
{
  "mcpServers": {
    "llmwiki": {
      "command": "/home/you/.local/bin/llmwiki",
      "args": ["mcp", "--watch"]
    }
  }
}
```

The server exposes `llmwiki_search`, `llmwiki_status`, `llmwiki_reindex`, and
`llmwiki_related`. Remove `--watch` if another process owns the watcher.

## Hermes plugin

Install Hermes first, then run:

```bash
git clone https://github.com/TechZeph/hermes-llmwiki-rag.git
cd hermes-llmwiki-rag
./install.sh --hermes
hermes gateway restart
```

The plugin uses the vault saved by `llmwiki init`. To select another vault:

```bash
hermes config set plugins.entries.llmwiki.settings.vault /path/to/vault
hermes gateway restart
```

You can also use `/llmwiki setup /path/to/vault` inside a Hermes session.
`/llmwiki status`, `/llmwiki reindex`, and `/llmwiki doctor` are available there
too. The in-host watcher is enabled by default after the first index; opt out
with `watch: false` in the plugin settings.

The plugin and MCP server make a non-blocking advisory update check when they
start. They never install updates automatically. In the Hermes plugin, disable
the network check with `update_check: false`; the MCP command does not currently
expose an update-check switch.

## Upgrade

From the cloned repository:

```bash
git pull --ff-only
./install.sh --no-init
```

For a Hermes installation:

```bash
./install.sh --hermes --no-init
hermes gateway restart
```

Your vault and generated index are retained.

## Uninstall

Disable the Hermes plugin first if it is installed:

```bash
hermes plugins disable llmwiki
rm -f ~/.hermes/plugins/llmwiki
```

Remove the standalone installation:

```bash
rm -f ~/.local/bin/llmwiki
rm -rf ~/.local/share/llmwiki/venv
```

Optional local data cleanup:

```bash
rm -f ~/.local/share/llmwiki/llmwiki.sqlite{,-wal,-shm}
rm -rf ~/.cache/fastembed
```

These commands do not modify the selected vault. Configuration remains at
`~/.config/llmwiki/config.toml` unless you remove it separately.

## Next steps

- [Configuration](configuration.md)
- [Commands and tools](tools.md)
- [Security and privacy](security.md)
- [Operations and offline use](operations.md)
- [Documentation index](README.md)