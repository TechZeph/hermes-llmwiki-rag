# Clean-environment install check transcript

Produced by `scripts/clean-install-check.sh` on 2026-09-01 against Hermes Agent 0.20.6 (source 2026.8.27), Python 3.14.4, from the built wheel in a fresh virtualenv with a temporary HERMES_HOME, an isolated XDG data dir and a throwaway three-page vault.

```
== work dir: /tmp/llmwiki-clean-ic0VWg
== hermes source: /home/ai-workstation/.hermes/hermes-agent (2a598aad1)
[1m* Creating isolated environment: venv+pip...[0m
[1m* Installing packages in isolated environment:[0m
  - setuptools>=68
  - wheel
[1m* Getting build dependencies for wheel...[0m
/tmp/build-env-lhwh872p/lib/python3.14/site-packages/setuptools/config/_apply_pyprojecttoml.py:82: SetuptoolsDeprecationWarning: `project.license` as a TOML table is deprecated
        By 2027-Feb-18, you need to update your project and remove deprecated calls
        or your builds will no longer be supported.
  corresp(dist, value, root_dir)
[1m* Installed build dependency versions:[0m
  - setuptools==84.0.0
  - wheel==0.48.0
[1m* Building wheel...[0m
/tmp/build-env-lhwh872p/lib/python3.14/site-packages/setuptools/config/_apply_pyprojecttoml.py:82: SetuptoolsDeprecationWarning: `project.license` as a TOML table is deprecated
        By 2027-Feb-18, you need to update your project and remove deprecated calls
        or your builds will no longer be supported.
  corresp(dist, value, root_dir)
== built hermes_llmwiki_rag-0.1.0-py3-none-any.whl
== note: hermes-agent editable install failed; doctor step will use the source tree on sys.path
== first index
done: seen=3 added=3 updated=0 removed=0 skipped=0 chunks: +4 ~0 -0 embeddings: built=4 rebuilt=0 errors=0
== integrity
ok schema 8
== doctor
llmwiki plugin loaded without a usable vault setting; tools will report the problem
Plugin Doctor: /home/ai-workstation/Workspace/repos/hermes-llmwiki-rag/hermes_plugin
  manifest: llmwiki 0.1.0 (backend)
  OK: runtime discovery, manifest parsing, import, and registration passed
  registrations: 4 tool(s), 2 hook(s)
== tool call through the Hermes plugin loader
status ok; top result: wiki/projects/demo/decisions.md | decision
related: ['wiki/sqlite-vec.md']
== PASS: clean install check completed
```

Notes: the editable install of hermes-agent into the fresh venv is optional and failed on its heavy dependencies; doctor and the tool call ran against the Hermes source tree on `sys.path`, which is the same code path the gateway uses. The "loaded without a usable vault setting" line comes from doctor's sandboxed HERMES_HOME, where no vault is configured by design.
