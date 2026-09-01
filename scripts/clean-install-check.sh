#!/usr/bin/env bash
# Clean-environment install check for the llmwiki Hermes plugin.
#
# Simulates a fresh machine: a new virtualenv, a temporary HERMES_HOME, the
# built wheel (not the source tree), an isolated FastEmbed cache, and a
# throwaway vault. Exercises install -> enable -> config -> first index ->
# doctor -> one tool call through Hermes' real plugin loader.
#
# Usage: scripts/clean-install-check.sh [path-to-hermes-agent-checkout]
# Requires network once for the embedding model unless FASTEMBED_CACHE_PATH
# already holds it. Prints a transcript suitable for docs/release evidence.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
HERMES_SRC="${1:-$HOME/.hermes/hermes-agent}"
WORK="$(mktemp -d -t llmwiki-clean-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

echo "== work dir: $WORK"
echo "== hermes source: $HERMES_SRC ($(cd "$HERMES_SRC" && git rev-parse --short HEAD 2>/dev/null || echo unknown))"

python3 -m venv "$WORK/venv"
PY="$WORK/venv/bin/python"
"$PY" -m pip install -q --upgrade pip build >/dev/null
( cd "$REPO" && "$PY" -m build --wheel --outdir "$WORK/dist" >/dev/null )
WHEEL="$(ls "$WORK"/dist/*.whl)"
echo "== built $(basename "$WHEEL")"
"$PY" -m pip install -q "$WHEEL[mcp]"
"$PY" -m pip install -q -e "$HERMES_SRC" >/dev/null 2>&1 || echo "== note: hermes-agent editable install failed; doctor step will use the source tree on sys.path"

# Throwaway vault with a few pages.
VAULT="$WORK/vault"; mkdir -p "$VAULT/wiki/projects/demo" "$VAULT/.obsidian"
cat > "$VAULT/wiki/sqlite-vec.md" <<'EOF'
# sqlite-vec

**Summary**: Vector search extension for SQLite.

---

## Storage

sqlite-vec stores float32 vectors in a vec0 virtual table.
EOF
cat > "$VAULT/wiki/projects/demo/decisions.md" <<'EOF'
# Demo decisions

## [2026-09-01] storage

We chose sqlite-vec over faiss because it keeps vectors inside SQLite.
EOF
cat > "$VAULT/wiki/projects/demo/log.md" <<'EOF'
# Demo log

## [2026-09-01] create | project log initialized
- created
EOF

export HERMES_HOME="$WORK/hermes-home"; mkdir -p "$HERMES_HOME/plugins"
export XDG_DATA_HOME="$WORK/xdg"
export FASTEMBED_CACHE_PATH="${FASTEMBED_CACHE_PATH:-$HOME/.cache/fastembed}"

# Directory plugin from the installed wheel (what `hermes plugins install` would produce).
PLUGIN_DIR="$("$PY" -c 'import hermes_plugin, os; print(os.path.dirname(hermes_plugin.__file__))')"
ln -s "$PLUGIN_DIR" "$HERMES_HOME/plugins/llmwiki"
cat > "$HERMES_HOME/config.yaml" <<EOF
plugins:
  enabled: [llmwiki]
  entries:
    llmwiki:
      settings:
        vault: $VAULT
EOF

echo "== first index"
LLMWIKI_VAULT="$VAULT" "$WORK/venv/bin/llmwiki" index 2>&1 | tail -1
echo "== integrity"
LLMWIKI_VAULT="$VAULT" "$WORK/venv/bin/llmwiki" integrity --json | "$PY" -c 'import json,sys; r=json.load(sys.stdin); print("ok" if r["ok"] else "FAILED", "schema", r["schema_version"])'

echo "== doctor"
( cd "$HERMES_SRC" && "$PY" - "$PLUGIN_DIR" <<'EOF'
import sys
from hermes_cli.plugin_dev import doctor_plugin
report = doctor_plugin(sys.argv[1])
print(report.format_text())
raise SystemExit(0 if report.ok else 1)
EOF
)

echo "== tool call through the Hermes plugin loader"
( cd "$HERMES_SRC" && "$PY" - <<'EOF'
import json, sys
sys.path.insert(0, ".")
from hermes_cli.plugins import PluginManager
from tools.registry import registry
pm = PluginManager(); pm.discover_and_load()
loaded = pm._plugins.get("llmwiki")
assert loaded and loaded.enabled and not loaded.error, getattr(loaded, "error", "not loaded")
status = json.loads(registry.get_entry("llmwiki_status").handler({}))
assert status["configured"] and status["integrity"]["ok"], status
res = json.loads(registry.get_entry("llmwiki_search").handler({"query": "why did we choose sqlite-vec over faiss?"}))
top = res["results"][0]
print("status ok; top result:", top["path"], "|", top["authority"])
assert top["path"] == "wiki/projects/demo/decisions.md", top
rel = json.loads(registry.get_entry("llmwiki_related").handler({"path": "wiki/projects/demo/decisions.md"}))
print("related:", [r["path"] for r in rel["related"]][:3])
EOF
)
echo "== PASS: clean install check completed"
