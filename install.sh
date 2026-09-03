#!/usr/bin/env bash
# llmwiki installer: virtualenv + package + first-run setup, optionally wired into Hermes.
#
#   ./install.sh                      # from a checkout: installs this source tree (editable)
#   ./install.sh --hermes             # also install into the Hermes venv, link + enable the plugin
#   ./install.sh --no-init            # skip `llmwiki init`
#   ./install.sh --dry-run            # print the plan, change nothing
#   ./install.sh -- /path/to/vault    # arguments after -- go to `llmwiki init`
#
# Environment: LLMWIKI_INSTALL_DIR (default ~/.local/share/llmwiki/venv),
#              LLMWIKI_BIN_DIR (default ~/.local/bin), PYTHON (default python3),
#              HERMES_HOME (default ~/.hermes), LLMWIKI_PACKAGE (default: this
#              checkout if run from one, else "llmwiki-rag[mcp]" from PyPI).
set -euo pipefail

INSTALL_DIR="${LLMWIKI_INSTALL_DIR:-$HOME/.local/share/llmwiki/venv}"
BIN_DIR="${LLMWIKI_BIN_DIR:-$HOME/.local/bin}"
PYTHON="${PYTHON:-python3}"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
WITH_HERMES=0; RUN_INIT=1; DRY_RUN=0; INIT_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --hermes) WITH_HERMES=1 ;;
    --no-init) RUN_INIT=0 ;;
    --dry-run) DRY_RUN=1 ;;
    --) shift; INIT_ARGS=("$@"); break ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '==> %s\n' "$*"; }
run() { if [ "$DRY_RUN" = 1 ]; then printf '    would run: %s\n' "$*"; else "$@"; fi; }

# --- locate the package source ------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || pwd)"
if [ -n "${LLMWIKI_PACKAGE:-}" ]; then
  PACKAGE_ARGS=("$LLMWIKI_PACKAGE"); SOURCE_MODE="explicit"
elif [ -f "$SCRIPT_DIR/pyproject.toml" ] && grep -q 'name = "llmwiki-rag"' "$SCRIPT_DIR/pyproject.toml"; then
  PACKAGE_ARGS=(-e "$SCRIPT_DIR[mcp]"); SOURCE_MODE="checkout ($SCRIPT_DIR)"
else
  PACKAGE_ARGS=("llmwiki-rag[mcp]"); SOURCE_MODE="PyPI"
fi

# --- python check --------------------------------------------------------------
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.11+ (https://www.python.org/downloads/) and re-run." >&2; exit 1
fi
PYVER="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "Python $PYVER found; llmwiki needs 3.11 or newer. Set PYTHON=/path/to/python3.11 and re-run." >&2; exit 1
fi
if ! "$PYTHON" -c 'import sqlite3; c=sqlite3.connect(":memory:"); c.execute("CREATE VIRTUAL TABLE t USING fts5(x)")' 2>/dev/null; then
  echo "This Python's sqlite3 lacks FTS5. Use python.org, Homebrew, or a distro python3 with FTS5." >&2; exit 1
fi

say "Python $PYVER at $(command -v "$PYTHON"); package source: $SOURCE_MODE"
say "virtualenv: $INSTALL_DIR ; launcher: $BIN_DIR/llmwiki"

# --- venv + package ------------------------------------------------------------
if [ ! -x "$INSTALL_DIR/bin/python" ]; then
  run mkdir -p "$(dirname "$INSTALL_DIR")"
  run "$PYTHON" -m venv "$INSTALL_DIR"
fi
VPY="$INSTALL_DIR/bin/python"
run "$VPY" -m pip install --quiet --upgrade pip
run "$VPY" -m pip install --quiet --upgrade "${PACKAGE_ARGS[@]}"
run mkdir -p "$BIN_DIR"
run ln -sfn "$INSTALL_DIR/bin/llmwiki" "$BIN_DIR/llmwiki"
case ":$PATH:" in *":$BIN_DIR:"*) ;; *) say "note: add $BIN_DIR to your PATH to run 'llmwiki' directly" ;; esac

# --- Hermes (optional) ---------------------------------------------------------
if [ "$WITH_HERMES" = 1 ]; then
  HPY="$HERMES_HOME/hermes-agent/venv/bin/python"
  if [ ! -x "$HPY" ]; then
    echo "Hermes venv not found at $HPY; install Hermes first (https://hermes-agent.nousresearch.com/docs/)" >&2; exit 1
  fi
  say "Hermes: installing the package into its venv and linking the plugin"
  run "$HPY" -m pip install --quiet --upgrade "${PACKAGE_ARGS[@]}"
  PLUGIN_DIR="$("$HPY" -c 'import hermes_plugin, os; print(os.path.dirname(hermes_plugin.__file__))' 2>/dev/null || true)"
  if [ "$DRY_RUN" = 1 ] && [ -z "$PLUGIN_DIR" ]; then PLUGIN_DIR="<hermes venv>/hermes_plugin"; fi
  run mkdir -p "$HERMES_HOME/plugins"
  run ln -sfn "$PLUGIN_DIR" "$HERMES_HOME/plugins/llmwiki"
  if command -v hermes >/dev/null 2>&1; then
    run hermes plugins enable llmwiki --no-allow-tool-override
  else
    say "note: 'hermes' not on PATH; run: hermes plugins enable llmwiki --no-allow-tool-override"
  fi
fi

# --- first-run setup -----------------------------------------------------------
if [ "$RUN_INIT" = 1 ]; then
  say "first-run setup (llmwiki init)"
  run "$INSTALL_DIR/bin/llmwiki" init "${INIT_ARGS[@]}"
else
  say "skipped init; run: $BIN_DIR/llmwiki init"
fi
say "done. Check everything with: $BIN_DIR/llmwiki doctor"
if [ "$WITH_HERMES" = 1 ]; then
  say "restart the Hermes gateway to load the plugin: hermes gateway restart"
fi
