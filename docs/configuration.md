# Configuration

Configuration is layered, lowest to highest precedence: package defaults →
user config file (`~/.config/llmwiki/config.toml`, written by `llmwiki init`
and `llmwiki config set`; `LLMWIKI_CONFIG` or `XDG_CONFIG_HOME` relocate it) →
`LLMWIKI_*` environment variables → host settings (Hermes plugin settings or
MCP/CLI flags) → per-call tool arguments. The user file holds `vault`, `db`,
`default_profile`, `retrieval_mode`, `embedding_model`. The Hermes plugin
falls back to the file's `vault` when `settings.vault` is unset.

## Core settings (`llmwiki.config.Settings`)

| setting | default | meaning |
|---|---|---|
| `vault_path` | required | Obsidian vault root; read-only; symlinks rejected |
| `db_path` | `$XDG_DATA_HOME/llmwiki/llmwiki.sqlite` | projection database |
| `embedding_model` | `BAAI/bge-small-en-v1.5` | FastEmbed model; dimension must match the vec0 schema (384) |
| `retrieval_mode` | `hybrid` | `dense`, `lexical`, or `hybrid` |
| `retrieval_top_k_dense` / `_lexical` | 50 / 50 | candidates per channel |
| `retrieval_top_k_final` | 10 | results returned |
| `rrf_k` | 20 | RRF constant (selected on dev) |
| `rrf_dense_weight` / `rrf_lexical_weight` | 1.0 / 1.0 | channel weights (selected jointly on v1 + v2 dev) |
| `max_chunks_per_document` | 3 | document diversification cap |
| `query_recipe` | `query-v2-bge-instruction` | query text recipe; never forces re-embeds |
| `reranker_enabled` / `reranker_model` / `rerank_candidates` | false / `BAAI/bge-reranker-base` / 30 | opt-in cross-encoder (Gate R failed) |
| `project_graph_expansion` / `_hops` / `_max_linked` | true / 1 / 40 | `project:<id>` admits linked curated pages |
| `graph_channel_enabled` / `_weight` / `_seed_documents` / `_max_neighbours` | false / 0.5 / 5 / 30 | linked-pages RRF channel (experiment) |
| `multiquery` | false | deterministic decomposition + fusion (experiment) |
| `recency_boost` | false | newest-first inside the authority tier for current-state intent (experiment) |
| `context_budget_tokens` / `context_per_document_tokens` / `context_max_excerpts` | 1500 / 600 / 8 | context builder budgets |

Environment variables for the CLI: `LLMWIKI_VAULT`, `LLMWIKI_DB`,
`LLMWIKI_LOG_LEVEL`, `LLMWIKI_LOG_FORMAT`, `FASTEMBED_CACHE_PATH`.

## Host settings (`llmwiki.service.ServiceConfig`)

Used by the Hermes plugin (`plugins.entries.llmwiki.settings.*`) and the
MCP server flags.

| key | default | meaning |
|---|---|---|
| `vault` | required | absolute vault path |
| `db` | XDG default | projection path |
| `default_profile` | `answer` | profile when a call omits one |
| `retrieval_mode` | `hybrid` | channel mix |
| `max_results` | 6 | excerpt cap per search (1–20) |
| `context_budget_tokens` | 1500 | context block budget |
| `rerank` | false | enable the cross-encoder |
| `allow_reindex` | true | permit incremental reindex via tool |
| `allow_full_rebuild` | false | permit `mode=full` (also needs `confirm=true`) |
| `stale_after_hours` | 24 | status flags the projection stale after this |
| `watch` / `watch_debounce_s` | false / 2 | in-host vault watcher (refuses a cold start) |
| `auto_inject` | false | opt-in `pre_llm_call` injection (needs a safety-certified gate) |
| `auto_inject_profile` | `answer` | fallback profile for automatic retrieval |
| `auto_inject_deadline_ms` | 1500 | internal deadline; timeout injects nothing (max 2000) |
| `auto_inject_budget_tokens` | 800 | injected context budget |

Hermes example (`~/.hermes/config.yaml`):

```yaml
plugins:
  enabled: [llmwiki]
  entries:
    llmwiki:
      settings:
        vault: /home/you/Workspace/vaults/clanker-vault
        watch: true
        max_results: 6
```

## Corpus profiles

| profile | admits |
|---|---|
| `answer` | curated `wiki/` pages; excludes `wiki/log.md`, project logs, route maps (`index*.md`), raw sources, clippings, root operational files |
| `project:<id>` | one project workspace minus its index and log, plus curated pages it links to (1 hop, ≤ 40) |
| `evidence` | `raw/**` and `Clippings/**` (ideas are labelled `idea`) |
| `history` | `wiki/log.md` and every project `log.md` |
| `all` | everything; diagnostics only |

Classification is path-derived and deterministic (`llmwiki.corpus`).
