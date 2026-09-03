# Configure llmwiki

Most users only need `llmwiki init`. It selects a vault, saves the choice, and
builds the first index. Use this page when you need a different profile, model,
resource limit, or host-specific setting.

Configuration is layered, lowest to highest precedence: package defaults →
user config file (`~/.config/llmwiki/config.toml`, written by `llmwiki init`
and `llmwiki config set`; `LLMWIKI_CONFIG` or `XDG_CONFIG_HOME` relocate it) →
`LLMWIKI_*` environment variables → host settings (Hermes plugin settings or
MCP/CLI flags) → per-call tool arguments. `llmwiki config set` manages `vault`,
`db`, `default_profile`, `retrieval_mode`, and `embedding_model`; resource
settings can be supplied through the environment variables below. The Hermes
plugin falls back to the file's `vault` when `settings.vault` is unset.

## Core settings (`llmwiki.config.Settings`)

| setting | default | meaning |
|---|---|---|
| `vault_path` | required | Obsidian vault root; read-only; symlinks rejected |
| `db_path` | `$XDG_DATA_HOME/llmwiki/llmwiki.sqlite` | projection database |
| `embedding_model` | `BAAI/bge-small-en-v1.5` | FastEmbed model; dimension must match the vec0 schema (384) |
| `retrieval_mode` | `hybrid` | `dense`, `lexical`, or `hybrid` |
| `retrieval_top_k_dense` / `_lexical` | 50 / 50 | candidates per channel |
| `retrieval_top_k_final` | 10 | results returned |
| `rrf_k` | 20 | reciprocal-rank-fusion constant selected by evaluation |
| `rrf_dense_weight` / `rrf_lexical_weight` | 1.0 / 1.0 | evaluated channel weights |
| `max_chunks_per_document` | 3 | document diversification cap |
| `query_recipe` | `query-v2-bge-instruction` | query text recipe; never forces re-embeds |
| `reranker_enabled` / `reranker_model` / `rerank_candidates` | false / `BAAI/bge-reranker-base` / 30 | opt-in cross-encoder (Gate R failed) |
| `project_graph_expansion` / `_hops` / `_max_linked` | true / 1 / 40 | `project:<id>` admits linked curated pages |
| `graph_channel_enabled` / `_weight` / `_seed_documents` / `_max_neighbours` | false / 0.5 / 5 / 30 | linked-pages RRF channel (experiment) |
| `multiquery` | false | deterministic decomposition + fusion (experiment) |
| `recency_boost` | false | newest-first inside the authority tier for current-state intent (experiment) |
| `context_budget_tokens` / `context_per_document_tokens` / `context_max_excerpts` | 1500 / 600 / 8 | context builder budgets |
| `resource_profile` | `balanced` | `conservative`, `balanced`, or `performance` embedding defaults |
| `embedding_batch_size` | profile default | optional fixed batch size (1–128) |
| `embedding_memory_budget_mb` | unset | optional advisory process-RSS ceiling in MiB |
| `embedding_min_available_mb` | 1024 | pause before a batch below this available-memory floor |
| `embedding_threads` | profile default | optional FastEmbed/ONNX thread limit |

Environment variables for the CLI include `LLMWIKI_VAULT`, `LLMWIKI_DB`,
`LLMWIKI_LOG_LEVEL`, `LLMWIKI_LOG_FORMAT`, `LLMWIKI_RESOURCE_PROFILE`,
`LLMWIKI_EMBEDDING_BATCH_SIZE`, `LLMWIKI_EMBEDDING_MEMORY_BUDGET_MB`,
`LLMWIKI_EMBEDDING_MIN_AVAILABLE_MB`, `LLMWIKI_EMBEDDING_THREADS`, and
`FASTEMBED_CACHE_PATH`.

## Host settings (`llmwiki.service.ServiceConfig`)

Used by the Hermes plugin (`plugins.entries.llmwiki.settings.*`). The MCP command
exposes a smaller subset as flags: vault, database, profile, result limit,
full-rebuild permission, and watcher control.

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
| `watch` / `watch_debounce_s` | true / 2 | in-host vault watcher (refuses a cold start) |
| `resource_profile` | `balanced` | embedding resource preset |
| `embedding_batch_size` | profile default | optional batch override (1–128) |
| `embedding_memory_budget_mb` | unset | optional advisory process-RSS ceiling in MiB |
| `embedding_min_available_mb` | 1024 | available-memory floor in MiB |
| `embedding_threads` | profile default | optional FastEmbed/ONNX thread limit |
| `update_check` / `update_check_timeout_s` | true / 2 | once per MCP/Hermes launch, check PyPI then GitHub Releases; advisory only, 1–10 s timeout per source |
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
        vault: /home/you/notes
        watch: true
        update_check: true
        max_results: 6
```

Update checks never install or modify the environment. Their result appears as
`update_check` in `llmwiki_status`; disable network checks with
`update_check: false`.

## Corpus profiles

| profile | admits |
|---|---|
| `answer` | curated `wiki/` pages; excludes `wiki/log.md`, project logs, route maps (`index*.md`), raw sources, clippings, root operational files |
| `project:<id>` | one project workspace minus its index and log, plus curated pages it links to (1 hop, ≤ 40) |
| `evidence` | `raw/**` and `Clippings/**` (ideas are labelled `idea`) |
| `history` | `wiki/log.md` and every project `log.md` |
| `all` | everything; diagnostics only |

Classification is path-derived and deterministic (`llmwiki.corpus`).

## Safe defaults

- Keep the core setting `retrieval_mode: hybrid`, the core setting
  `reranker_enabled: false`, and the host setting `auto_inject: false` unless
  you are deliberately evaluating another configuration.
- Keep `resource_profile: balanced` on a typical workstation. Use
  `conservative` on a memory-constrained machine and `performance` only when
  you have measured available headroom.
- Resource settings are advisory. For a hard operating-system memory or CPU
  limit on Linux, see [Operations](operations.md).
- Run only one watcher against a generated index.

## Next steps

- [Install](install.md)
- [Commands and tools](tools.md)
- [Security and privacy](security.md)
- [Documentation index](README.md)
