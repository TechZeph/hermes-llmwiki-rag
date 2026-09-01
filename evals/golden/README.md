# Golden question sets

Real-vault retrieval questions used by `llmwiki eval`. Each file is a JSON
object: `{"corpus": "<vault name>", "version": "<set version>", "questions": [...]}`.

Question schema (all keys required unless noted):

```json
{
  "id": "cs-001",
  "category": "current-state | decision | exact-term | concept | evidence | chronology | ambiguity | no-answer",
  "split": "dev | heldout",
  "query": "natural-language question as a user would type it",
  "profile": "answer | project:<id> | evidence | history",
  "authority_class": "current-state | decision | durable | evidence | log | idea | none",
  "mode": "retrieve | abstain",
  "relevant": [
    {"path": "wiki/some-page.md", "sections": ["Optional heading text", "..."]}
  ],
  "notes": "optional: why this question exists / what makes it hard"
}
```

Rules:

- `relevant[].path` is vault-relative with forward slashes and must exist on disk.
- `sections` is optional; when present, each entry must be a heading text that
  appears in that file (exact text after the `#` marks). An empty list or absent
  key means any chunk of the document counts.
- `mode: abstain` questions have `relevant: []` and `authority_class: none`.
- `ambiguity` questions list every page a reasonable reader could mean.
- Roughly 30% of every category is `heldout`; the rest is `dev`.
