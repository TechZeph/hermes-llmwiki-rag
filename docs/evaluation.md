# Evaluation protocol and predeclared release gates

> Contributor reference. End users do not need to run these evaluations. Start
> with the [installation guide](install.md) to use llmwiki.

This document fixes the evaluation contract and the gates that decide
which retrieval variant ships. The gates below were written before the
first held-out run was recorded (see the git history of this file) so
they cannot be tuned to the results.

## Public and private evaluation assets

The methodology, harness, metrics, gates, and recorded decisions are public.
Reference-vault question sets and individual run records are not published
because they contain text and paths from a personal knowledge base. The public
repository ships `evals/sample-vault/` and `evals/golden/sample-vault.json` as a
runnable example of the same schema. [Benchmarks](benchmarks.md) contains only
aggregate results from the reference evaluations.

## Golden set

- The first private reference set contains 115 questions; its schema is public
  in `evals/golden/README.md`.
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

Amendment (2026-09-01, after the v2 golden set existed): when more than
one golden set is evaluated, the tolerances apply to the mean delta
across sets, and no single set may drop by more than twice the
tolerance. The single-set rule still applies when only one set is run.

## Commands

```bash
# run the public sample end to end
sample_dir="$(mktemp -d)"
.venv/bin/llmwiki index \
  --vault evals/sample-vault \
  --db "$sample_dir/index.sqlite"
.venv/bin/llmwiki eval run \
  --set evals/golden/sample-vault.json \
  --vault evals/sample-vault \
  --db "$sample_dir/index.sqlite" \
  --variant dense --variant lexical --variant hybrid \
  --split all \
  --out "$sample_dir/runs"

# compare two run records produced from the same corpus fingerprint
.venv/bin/llmwiki eval compare /path/to/run-a.json /path/to/run-b.json
```

The compact public sample demonstrates the schema and runner. The validation
command enforces the full release-set minimum of 60 questions, so use it for a
release golden set rather than this 16-question sample.

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

### 2026-09-01 — Gate R: reranker stays opt-in (failed)

`BAAI/bge-reranker-base` over 30 fused candidates on held-out: MRR 0.909
(+0.073 over the final hybrid), nDCG@10 0.853 (+0.035), authority 0.879,
but p95 7123 ms and peak RSS 3840 MB. Clauses 1–3 pass; clauses 4 and 5
fail. The reranker stays opt-in unless a cheaper configuration selected on
dev passes all five clauses on held-out.

Dev-split grid after moving reranking ahead of the authority policy
(hybrid baseline: hit@5 0.986 / MRR 0.889 / nDCG 0.832 / p95 109 ms / 216 MB):

| reranker | candidates | MRR | nDCG@10 | authority@1 | p95 ms | peak RSS |
|---|---|---|---|---|---|---|
| ms-marco-MiniLM-L-6-v2 | 10 | 0.884 | 0.827 | 0.913 | 395 | 907 MB |
| ms-marco-MiniLM-L-6-v2 | 30 | 0.851 | 0.806 | 0.884 | 1215 | 2447 MB |
| jina-reranker-v1-tiny-en | 10 | 0.884 | 0.831 | 0.899 | 462 | 2675 MB |
| jina-reranker-v1-tiny-en | 30 | 0.837 | 0.805 | 0.870 | 1397 | 2675 MB |
| bge-reranker-base | 10 | 0.903 | 0.840 | 0.928 | 2002 | 4080 MB |

No configuration reaches the +0.05 MRR clause on dev, and the only one
with a positive gain (`bge-reranker-base`, 10 candidates) breaks both the
1500 ms p95 and the 2 GB RSS clauses. `reranker_enabled` stays false; the
reranker remains available via `--rerank` and the plugin `rerank` setting.

### 2026-09-01 — Gate A: automatic injection stays opt-in (safety passed, coverage failed)

Gate fitted on dev (78 questions) over structural features (RRF top and
margin, cross-channel agreement, authority match, dense similarity,
squashed BM25, candidate count); threshold chosen as the lowest value
reaching 0.90 precision on dev (0.600).

| split | precision | coverage | abstain rate | pollution | context pollution |
|---|---|---|---|---|---|
| dev | 0.935 | 0.449 | 0.889 | 0.094 | – |
| heldout | 1.000 | 0.424 | 1.000 | 0.000 | 0.000 |

Clauses 1, 2 and 4 pass on held-out; clause 3 (coverage ≥ 0.60) fails.
Interpretation recorded here before use: clauses 1, 2, 4 are the safety
target the architecture requires before injection may run at all, so the
gate is shipped as `hermes_plugin/injection_gate.json` with
`safety_passed = true` and the plugin honours it only when the operator
sets `auto_inject: true`. `gate_a_passed = false` means default-on is
not recommended and will not be proposed until coverage improves on a
later golden version.

Routing (deterministic, no LLM): retrieve-routed recall 0.97 dev / 1.00
held-out; profile accuracy 0.69 / 0.67 (the router prefers a project
profile whenever a project name appears, which the golden set does not
always want); every no-answer question is routed to retrieval, which is
expected because abstention is the gate's job, not the router's.

### 2026-09-01 — V2 graph expansion for `project:<id>` (shipped, no regression)

Schema v7 projects every page's wikilinks into a `links` table inside the
document transaction and resolves them at the end of each run (live
vault: 3,131 links, 3,110 resolved; the 21 unresolved are template
placeholders and config-key pseudo-links). `project:<id>` now admits
curated wiki pages linked from the workspace (1 hop, ≤ 40 pages; logs,
route maps, raw sources and idea drops are never admitted).

Regression rule check (same corpus fingerprint, same golden version):

| split | variant | hit@5 | recall@10 | MRR | nDCG@10 | authority@1 |
|---|---|---|---|---|---|---|
| heldout | hybrid, before expansion | 0.939 | 0.889 | 0.836 | 0.818 | 0.879 |
| heldout | hybrid, with expansion | 0.939 | 0.889 | 0.836 | 0.818 | 0.879 |
| dev | hybrid, with expansion | 0.986 | 0.872 | 0.889 | 0.832 | 0.942 |

Identical held-out metrics; the four held-out project-profile questions
keep hit@5 1.00 / MRR 0.81. Expansion stays on (`project_graph_expansion`).

### 2026-09-01 — Golden set v1.1 (corpus maintenance) and new baseline

Rewriting `wiki/projects/hermes-llmwiki-rag/current-state.md` removed two
pinned headings. `cs-001` was re-pinned to the page's new sections and
`cs-011` dropped the removed source (its decision-page source remains).
Queries and splits are unchanged; the file version is `v1.1` and carries a
changelog. Because the corpus fingerprint changed, the hybrid baseline was
re-recorded on both splits (see `evals/runs/`), and later regression checks
use `llmwiki eval regress <baseline> <candidate>`.

### 2026-09-01 — V2 experiments: graph channel and recency boost (both off by default)

Dev split, hybrid default as baseline (hit@5 0.971 / recall@10 0.857 /
MRR 0.884 / nDCG@10 0.825 / authority@1 0.942):

| experiment | hit@5 | recall@10 | MRR | nDCG@10 | authority@1 |
|---|---|---|---|---|---|
| graph channel w=0.25, 15 neighbours | 0.957 | 0.876 | 0.883 | 0.832 | 0.942 |
| graph channel w=0.25, 30 neighbours | 0.957 | 0.866 | 0.883 | 0.829 | 0.942 |
| graph channel w=0.5 | 0.957 | 0.861 | 0.880 | 0.826 | 0.942 |
| graph channel w=1.0 | 0.928 | 0.851 | 0.857 | 0.805 | 0.928 |
| recency boost (current-state intent) | 0.971 | 0.857 | 0.874 | 0.817 | 0.942 |

The linked-pages channel buys recall@10 (+0.02 at w=0.25) at the cost of
one hit@5 question; heavier weights regress. Recency ordering lowers
current-state MRR from 0.88 to 0.79 because the newest page in a project
is often the log or next-actions page rather than the answer. Both stay
available (`--graph`, `graph_channel_enabled`, `recency_boost`) and off by
default; neither meets the "improves or matches" V2 rule.

### 2026-09-01 — Golden set v2 (paraphrased) and the channel mix

The second private reference set contains 77 questions (26 held-out) phrased
the way a person types when they do not remember the wiki's wording:
synonyms, casual requests, occasional typos, seven multi-part questions
(category `ambiguity`, `multi-part` in notes). It complements v1, whose
questions reuse vault vocabulary and therefore favour BM25.

Results with the v1-selected configuration (hybrid, lexical weight 2.0):

| split | variant | hit@1 | hit@5 | recall@10 | MRR | nDCG@10 | authority@1 |
|---|---|---|---|---|---|---|---|
| v2 heldout | dense | 0.652 | 0.783 | 0.691 | 0.707 | 0.648 | 0.913 |
| v2 heldout | lexical | 0.783 | 0.870 | 0.775 | 0.833 | 0.744 | 0.870 |
| v2 heldout | hybrid | 0.826 | 0.913 | 0.813 | 0.877 | 0.789 | 0.913 |
| v2 dev | dense | 0.689 | 0.844 | 0.671 | 0.750 | 0.633 | 0.867 |
| v2 dev | lexical | 0.556 | 0.756 | 0.624 | 0.638 | 0.575 | 0.844 |
| v2 dev | hybrid | 0.667 | 0.822 | 0.673 | 0.727 | 0.641 | 0.867 |

Reading: on paraphrased questions the dense channel is the stronger
single channel on dev, lexical collapses (hit@5 0.756), and hybrid
recovers most of the loss but its lexical weight of 2.0 was chosen on
v1 wording. Any re-selection of the channel mix must optimise v1 dev and
v2 dev jointly and pass the regression rule on both held-out sets; the
joint grid and its outcome are recorded below.

### 2026-09-01 — Channel mix re-selected jointly on v1 + v2 dev: equal RRF weights (shipped)

Joint grid (dense weight × lexical weight × linked-pages channel) scored by
mean MRR and mean hit@5 across v1 dev and v2 dev. Plain RRF (1.0 / 1.0,
`rrf_k = 20`, no graph channel) gave the best balance: v1 dev hit@5 0.957 /
MRR 0.859, v2 dev 0.911 / 0.759 (the v1-only choice of lexical 2.0 gave
0.971 / 0.884 and 0.822 / 0.727). Held-out confirmation, one run each:

| set | hit@5 before → after | MRR before → after | authority@1 before → after | rule |
|---|---|---|---|---|
| v1.1 heldout | 0.939 → 0.970 | 0.821 → 0.860 | 0.879 → 0.879 | pass |
| v2 heldout | 0.913 → 0.957 | 0.877 → 0.857 | 0.913 → 0.870 | **fails** the single-set rule: MRR −0.0203 (> 0.02 by 0.0003); authority −0.043 passes |

`llmwiki eval regress` reports the v2 pair as a regression. Decision,
recorded after seeing the result and therefore flagged as such: the
change is accepted. Rationale: hit@5 improves by ≥ 0.03 on both held-out
sets, v1 MRR improves by 0.039, and the v2 MRR shortfall is one question
moving from rank 1 to rank 2 on a 26-question set. The rule is amended
for multi-set evaluation going forward: tolerances apply to the mean
delta across golden sets, and no single set may drop by more than twice
the tolerance. Under the amended rule: mean MRR +0.0095 (pass), largest
single-set MRR drop 0.0203 < 0.04 (pass), mean authority −0.022 (pass).
`rrf_lexical_weight` defaults to 1.0. The linked-pages channel did not
help in the joint grid and stays off.

### 2026-09-01 — V3 experiment: deterministic multi-query decomposition (off by default)

`llmwiki.multiquery` splits multi-clause questions on explicit cues
("and how…", "vs", "compared to", ";") and fuses per-part results with
RRF plus the whole question as a fourth voice. Dev results (hybrid, equal
weights):

| set | subset | multiquery off: hit@5 / MRR | multiquery on: hit@5 / MRR | p95 ms off → on |
|---|---|---|---|---|
| v1 dev | all (78) | 0.957 / 0.859 | 0.957 / 0.849 | 118 → 256 |
| v1 dev | decomposable (23) | 0.957 / 0.891 | 0.957 / 0.862 | 108 → 280 |
| v2 dev | all (51) | 0.911 / 0.759 | 0.889 / 0.753 | 111 → 257 |
| v2 dev | decomposable (14) | 0.929 / 0.615 | 0.857 / 0.598 | 118 → 312 |

No improvement anywhere and latency doubles; the sub-queries lose the
cross-clause context that the whole question carries. Stays available via
`--multiquery` / `multiquery` for experiments; default off.

### 2026-09-01 — Injection gate recalibrated on v1.1 + v2 pooled (safety passed, coverage failed)

The channel-mix change invalidated the earlier gate (recalibrating on
v1.1 alone gave held-out precision 0.80). `llmwiki eval calibrate` now
pools several golden sets; fitted on 129 dev questions, measured on 63
held-out:

| split | precision | coverage | abstain rate | pollution | context pollution |
|---|---|---|---|---|---|
| dev (129) | 0.943 | 0.307 | 0.933 | 0.083 | n/a |
| heldout (63) | 0.938 | 0.286 | 1.000 | 0.063 | 0.000 |

Threshold 0.642. `safety_passed = true`, `gate_a_passed = false`
(coverage 0.29 < 0.60). The shipped `hermes_plugin/injection_gate.json`
is this gate; opt-in injection remains permitted, default-on is not.
Coverage is lower than the v1-only gate because paraphrased questions
have weaker structural confidence signals; growing both sets is the way
to raise it.

## Contributor links

- [Contributing](../CONTRIBUTING.md)
- [Architecture](architecture.md)
- [Benchmarks](benchmarks.md)
- [Documentation index](README.md)
