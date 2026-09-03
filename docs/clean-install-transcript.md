# Pre-release installation verification

> Release-engineering record for contributors. This is not an installation
> guide; users should follow [Install llmwiki](install.md).

The pre-release build targeting version 0.1.0 was exercised in an isolated
environment on 2026-09-01. The check built the wheel, installed it into a new virtual
environment, indexed the synthetic sample vault, ran integrity and doctor
checks, loaded the Hermes plugin through Plugin Doctor, and invoked a registered
tool through the Hermes plugin loader.

## Verified results

- Wheel built: `hermes_llmwiki_rag-0.1.0-py3-none-any.whl`.
- Synthetic first index: 3 documents, 4 chunks, 4 embeddings, 0 errors.
- Integrity: schema v8; document, chunk, vector, FTS, and link counts consistent.
- Plugin manifest: `llmwiki` 0.1.0 loaded and registered successfully.
- Tool call: `llmwiki_status` returned a configured, healthy projection.
- Source tree after the check: clean.

The check used the Hermes source tree on `sys.path` for Plugin Doctor rather
than installing Hermes and all of its optional dependencies into the temporary
environment. It therefore verifies the plugin registration path, not a clean
third-party Hermes installation. A final install from the published release is
still required before claiming external-install verification.

The repeatable check is `scripts/clean-install-check.sh`. CI separately runs the
test suite, package build, static checks, and Plugin Doctor across the supported
matrix.

## Related documentation

- [Contributing](../CONTRIBUTING.md)
- [Evaluation](evaluation.md)
- [Architecture](architecture.md)