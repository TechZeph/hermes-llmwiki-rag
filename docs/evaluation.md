# Evaluation protocol and predeclared release gates

This document fixes the evaluation contract and the gates that decide
which retrieval variant ships. The gates below were written before the
first held-out run was recorded (see the git history of this file) so
they cannot be tuned to the results.

## Golden set

- File: `evals/golden/clanker-vault-v1.json`, schema in `evals/golden/README.md`.
- 115 real-vault questions across eight categories: current-state,
  decision, exact-term, concept, evidence, chronology, ambiguity,
  no-answer. Every category has a `dev` and a `heldout` split
  (37 held-out questions overall).
- Each question names the corpus profile the caller would use, the
  authority class of the expected answer, the retrieve/abstain mode,
  and every page (optionally pinned to headings) that legitimately
  answers it.
- Validate with `llmwiki eval validate --set ... --vault ...`. Paths and
  headings are checked against the live vault so a renamed page fails
  validation instead of silently lowering recall.

Rules of use:

- Tune on `dev` only. Report `heldout` numbers for every decision below.
- Do not edit a question after looking at held-out results for it. Add
  a new question instead and bump the set version.
- Ambiguity questions list every plausible page; `hit@k` credits any
  of them, `recall@k` credits all of them.

## Metrics recorded per run

Every run record (`evals/runs/*.json`) stores: variant, split, git SHA
(with `-dirty` when the tree is modified), corpus fingerprint (hash of
every indexed path + content hash), document/chunk/vector/FTS counts,
`projection_meta` (recipe versions, model, dimension, FastEmbed
provenance), the retrieval settings used, Python/SQLite/package
versions, peak RSS, and per-question outcomes.

Aggregates (overall and per category):

| Metric | Definition |
|---|---|
| `hit@k` | fraction of retrieve questions with at least one relevant page in the top k unique documents |
| `recall@k` | mean fraction of a question's relevant pages found in the top k |
| `MRR` | mean reciprocal rank of the first relevant page |
| `nDCG@10` | binary-gain nDCG over unique documents |
| `authority_accuracy_top1` | top-1 result's authority class satisfies the question's authority class |
| `authority_any_top3` | same, anywhere in the top 3 |
| `duplicate_concentration` | share of the top 10 chunks taken by the single most frequent document |
| `citation_fidelity` | share of returned chunks whose path resolves in the vault, whose stored hash matches the text, and whose section matches its breadcrumb |
| `conflict_rate` | share of retrieve questions with a provenance conflict label |
| `abstain_*` / `retrieve_*` | mean top-score features on abstain vs retrieve questions, recorded for calibration; not a metric of correctness until a gate exists |
| `latency_p50_ms`, `latency_p95_ms` | end-to-end retrieval latency per question, excluding process start and model load |

Section-pinned questions count a document as hit only when a returned
chunk's section or breadcrumb matches a pinned heading.

## Predeclared gates

All comparisons use the same held-out split, the same golden version,
the same corpus fingerprint, and `top_k = 10`.

### Gate H: hybrid becomes the default retrieval mode

Hybrid (dense + BM25 with RRF, `rrf_k = 60`, 50 candidates per channel)
becomes the default if, on held-out:

1. `hit@5` is at least the best single channel minus 0.02, and
2. `MRR` is at least the best single channel minus 0.02, and
3. `authority_accuracy_top1` is not lower than the best single channel by more than 0.05, and
4. `latency_p95_ms` is at most 250 ms on the reference workstation.

If hybrid fails any clause, the best single channel stays the default
and the failure is recorded in the vault decisions page.

### Gate R: a cross-encoder reranker ships enabled

`hybrid+rerank` (reranking the top 30 fused candidates) ships with
`reranker_enabled = true` only if, on held-out and relative to the
default chosen by Gate H:

1. `MRR` improves by at least 0.05 absolute, and
2. `nDCG@10` improves by at least 0.03 absolute, and
3. `authority_accuracy_top1` does not fall by more than 0.02, and
4. `latency_p95_ms` is at most 1500 ms, and
5. peak RSS increases by at most 2 GB over the default variant.

Otherwise the reranker stays available behind `--rerank` / settings and
is reported as an experiment.

### Gate A: automatic injection (V1.1) may be enabled

Automatic `pre_llm_call` injection may default to on only if a
calibrated gate, fitted on `dev` and measured on `heldout`, achieves:

1. injection precision of at least 0.90 (injected turns whose top result is relevant), and
2. abstain rate on no-answer questions of at least 0.80, and
3. coverage of at least 0.60 on retrieve questions, and
4. context-pollution rate (injected but no relevant page in the context) of at most 0.10, and
5. the hook's internal deadline never exceeds 2 s.

Until then automatic injection stays off by default and is opt-in per
profile.

### Regression rule

Any later change to chunking, recipes, model, fusion, or authority
policy re-runs the full matrix. A change is a regression if held-out
`hit@5` or `MRR` drops by more than 0.02 or `authority_accuracy_top1`
drops by more than 0.05 against the last accepted run for the same
corpus fingerprint.

## Commands

```bash
# validate the set against the live vault
.venv/bin/llmwiki eval validate --set evals/golden/clanker-vault-v1.json --vault ~/Workspace/vaults/clanker-vault

# run the matrix on held-out
.venv/bin/llmwiki eval run --set evals/golden/clanker-vault-v1.json \
  --vault ~/Workspace/vaults/clanker-vault \
  --variant dense --variant lexical --variant hybrid --variant hybrid+rerank \
  --split heldout

# compare recorded runs
.venv/bin/llmwiki eval compare evals/runs/*.json
```

## Recorded decisions

### 2026-09-01 — Gate H: hybrid is the default (passed)

Dev-split selection (78 questions): query recipe `query-v2-bge-instruction`,
`rrf_k = 20`, channel weights dense 1.0 / lexical 2.0, 50 candidates per
channel, 3 chunks per document. Grid results are in the session log; the
best plain-RRF configuration (`k = 60`, equal weights) reached hit@5 0.971 /
MRR 0.876 on dev against lexical's 0.971 / 0.894, and the selected
configuration reached 0.986 / 0.889.

Held-out (37 questions, one run after selection):

| variant | hit@5 | recall@10 | MRR | nDCG@10 | authority@1 | p95 ms |
|---|---|---|---|---|---|---|
| dense | 0.848 | 0.786 | 0.773 | 0.726 | 0.848 | 77 |
| lexical | 0.939 | 0.864 | 0.831 | 0.815 | 0.879 | 6 |
| hybrid | 0.939 | 0.889 | 0.836 | 0.818 | 0.879 | 83 |

Gate H clauses: hit@5 0.939 ≥ 0.919 ✔; MRR 0.836 ≥ 0.811 ✔; authority 0.879
vs 0.879 ✔; p95 83 ms ≤ 250 ms ✔. `retrieval_mode = "hybrid"` is the default.

Observation for later work: BM25 alone is unusually strong on this vault
because the golden questions reuse the vault's own terminology; the
dense channel mainly adds recall@10 and robustness on concept phrasing.

### 2026-09-01 — Gate R: reranker (see below once the resource grid completes)

`BAAI/bge-reranker-base` over 30 fused candidates on held-out: MRR 0.909
(+0.073 over the final hybrid), nDCG@10 0.853 (+0.035), authority 0.879,
but p95 7123 ms and peak RSS 3840 MB. Clauses 1–3 pass; clauses 4 and 5
fail. The reranker stays opt-in unless a cheaper configuration selected on
dev passes all five clauses on held-out.
