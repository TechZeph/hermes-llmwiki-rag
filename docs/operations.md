# Local data and offline operation

## Projection sensitivity

`llmwiki` treats the SQLite projection as sensitive local data. It contains chunk text, metadata (including frontmatter-derived values), hashes, embeddings, model/recipe metadata, and diagnostic absolute paths. Markdown in the selected vault remains canonical; the projection is rebuildable.

Do not place the default projection inside the vault. The default is `$XDG_DATA_HOME/llmwiki/llmwiki.sqlite`, falling back to `~/.local/share/llmwiki/llmwiki.sqlite`.

On POSIX, opening a projection creates or repairs its parent directory as `0700` and its database, `-wal`, and `-shm` files as `0600`. SQLite sidecars are created under a restrictive umask and permissions are rechecked after WAL is enabled. On non-POSIX systems, `llmwiki` emits a warning rather than claiming equivalent ACL protection; configure an owner-only directory/ACL yourself.

## Model provisioning and offline use

The normal index/search path uses FastEmbed locally. The first construction of `FastEmbedEmbedder` may acquire a model through FastEmbed if it is absent from its cache; ordinary use after the model is cached does not intentionally make network requests.

Provision before an offline deployment:

1. Choose a cache directory that is private to the runtime account and set `FASTEMBED_CACHE_PATH` to it.
2. On a connected machine, run an index using the exact configured model. This causes FastEmbed to provision the model cache.
3. Copy or preserve that cache with the deployment, then disconnect the runtime environment and run `llmwiki index` or `llmwiki search` as an offline smoke test.
4. Keep the same FastEmbed package version and cache contents for a reproducible projection. A projection stores the configured model name, effective vector dimension, FastEmbed package version, and FastEmbed registry artifact source.

FastEmbed's registry currently does not expose an immutable artifact checksum through this package interface. The stored provenance is therefore not a byte-level model attestation. Preserve the provisioned cache and record its checksum in deployment tooling when that assurance is required.

When a model or embedding-recipe change requires a full re-embedding, `llmwiki` performs the vector replacement in one SQLite transaction. If embedding fails or the process is interrupted, the last complete vector set is retained; a successful migration may hold the projection write lock for its full duration.

## Logging and telemetry

The package has no telemetry client and writes no service logs by default. CLI operational logs go to stderr; avoid `DEBUG` in shared terminals or CI logs. Default operational events must not include query text, conversation history, chunk bodies, frontmatter, embeddings, or vault absolute paths. Use vault-relative identifiers and error classes for future diagnostics.

There is currently no persisted query history, conversation history, telemetry spool, or evaluation artifact store. If one is introduced, it must have an explicit retention setting, an owner-visible location, and a scoped deletion operation before it is enabled by default.

## Backup, restore, and deletion

There is no `backup` or `delete-projection` command yet. Do not copy a live SQLite database file while WAL is active: a plain copy can omit committed WAL content. The safe current backup is the canonical Markdown vault plus the exact model cache if offline operation is needed; rebuild the projection with `llmwiki index --mode full` after restore.

To discard the current projection, stop all `llmwiki` processes and remove the database together with its sibling `-wal` and `-shm` files. Do not remove the vault or model cache unless that is separately intended. File removal is not a promise of forensic erasure: recovery depends on the storage device, filesystem, encryption, and retention/snapshot policy.

Future backup and deletion commands must expose separate scopes for the projection, model cache, logs, and any telemetry/evaluation artifacts, and a backup must use a consistent SQLite snapshot that incorporates WAL content.
