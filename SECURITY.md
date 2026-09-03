# Security policy

## Supported versions

Security fixes are applied to the latest published release. Before the first tagged release, fixes are applied to `main`.

## Report a vulnerability

Do not open a public issue for a suspected vulnerability or include sensitive vault content in a report.

Email `hello@techzeph.co.uk` with the subject `llmwiki security report`. Include:

- the affected version or commit;
- the operating system and installation method;
- a minimal reproduction that does not expose personal vault data;
- the expected impact;
- any suggested mitigation.

Reports will be acknowledged as soon as practical. No public disclosure timeline
is promised until the report has been reproduced and a safe fix is available.

## Scope

Reports about vault path containment, unintended writes, projection permissions, query or content leakage, unsafe retrieved-context handling, dependency execution, or host/plugin boundaries are in scope.

General bugs and feature requests belong in [GitHub Issues](https://github.com/TechZeph/llmwiki-rag/issues).

See [docs/security.md](docs/security.md) for the trust and privacy model.
