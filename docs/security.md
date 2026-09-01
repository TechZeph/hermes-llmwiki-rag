# Security and privacy model

## Trust boundaries

- **The vault is canonical and read-only.** The package never writes,
  moves, or deletes vault files. Discovery prunes symlinked directories,
  skips symlinked files, and re-proves containment inside the configured
  vault immediately before every stat and parse.
- **The projection is sensitive local data.** It holds chunk text,
  frontmatter, paths, hashes and vectors. It is created outside the vault
  with `0700` directory and `0600` file permissions on POSIX; other
  platforms get a warning, not a false guarantee. See `operations.md` for
  backup and deletion limits.
- **Plugins are in-process trusted code.** Installing the Hermes plugin or
  running the MCP server executes this package inside the host. Pin the
  release you audited; `hermes plugins doctor --ci` validates the
  registration contract, not the code.
- **Models are a one-time supply-chain step.** FastEmbed downloads
  `BAAI/bge-small-en-v1.5` once; runtime is offline afterwards. The
  projection records the FastEmbed version and registry artifact source,
  not a byte-level checksum; preserve the cache for reproducibility.

## Retrieved content is evidence, never instructions

Every context block is wrapped in fixed delimiters and each excerpt
carries a frame naming its source, section, authority class, source kind
and retrieval mode. Inside retrieved text, runs of three or more angle
brackets and any `[excerpt …]`/`[/excerpt …]` markers are rewritten so
retrieved Markdown cannot close or forge the envelope. Six
prompt-injection fixtures in the test suite assert this on every change.

Raw sources, clippings, idea drops and logs are labelled as such
(`evidence`, `idea`, `log`) and only reachable through explicit profiles;
the default `answer` profile serves curated wiki pages only.

## What is never logged or persisted by this package

- Query text and conversation history.
- Retrieved chunk bodies in operational logs.
- Absolute filesystem paths in tool responses (status shows the vault
  basename; errors replace the vault path with `<vault>`).

Injection decisions are kept in memory (last 50) with features and
outcomes only. There is no telemetry.

## Automatic injection (opt-in)

`pre_llm_call` is registered but inert by default. When enabled it uses
only the current user message, a deterministic router, a bounded internal
deadline (≤ 2 s) and a calibrated logistic gate whose shipped metrics must
show `safety_passed = true` (held-out precision ≥ 0.90, abstain rate ≥
0.80, pollution ≤ 0.10). Anything else returns nothing. Two caveats:

1. Hermes appends injected context to the user message and may persist
   it in an `api_content` sidecar for replay; enabling injection means
   retrieved excerpts can appear in Hermes session storage.
2. The current gate is conservative (coverage 0.42), so default-on is not
   recommended and is not offered.

## Reporting

Security issues: open a private report on the GitHub repository rather
than a public issue.
