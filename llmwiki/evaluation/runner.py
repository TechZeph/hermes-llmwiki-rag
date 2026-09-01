"""Run a golden set through a retriever variant and persist a reproducible record."""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import sqlite3
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from ..models import Candidate, RetrievalResult
from ..retrieval import Retriever
from .golden import GoldenSet, Question
from .metrics import (
    Aggregate,
    QuestionOutcome,
    aggregate,
    corpus_fingerprint,
    per_category,
    score_question,
)

VARIANTS: tuple[str, ...] = ("dense", "lexical", "hybrid", "hybrid+rerank")


@dataclass(slots=True)
class RunRecord:
    variant: str
    split: str
    golden_set: str
    golden_version: str
    question_count: int
    started_at: str
    finished_at: str
    git_sha: str
    corpus_fingerprint: str
    document_count: int
    chunk_count: int
    vector_count: int
    fts_count: int
    projection_meta: dict[str, str]
    settings: dict[str, Any]
    environment: dict[str, str]
    peak_rss_mb: float
    overall: dict[str, Any]
    by_category: dict[str, dict[str, Any]]
    outcomes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def git_sha(repo: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo or Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo or Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # pragma: no cover - git absent
        return "unknown"


def projection_snapshot(conn: sqlite3.Connection) -> dict[str, Any]:
    rows = conn.execute("SELECT path, content_hash FROM documents").fetchall()
    meta = {
        str(k): str(v) for k, v in conn.execute("SELECT key, value FROM projection_meta").fetchall()
    }
    counts = {
        "documents": int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]),
        "chunks": int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]),
        "vectors": int(conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]),
        "fts": int(conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]),
    }
    return {
        "fingerprint": corpus_fingerprint([(str(p), str(h)) for p, h in rows]),
        "meta": meta,
        "counts": counts,
    }


def _peak_rss_mb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(kb / 1024.0, 1)


def _citation_checks(vault: Path | None, candidates: Sequence[Candidate]) -> list[bool]:
    """Structural citation fidelity: path resolves and the chunk hash matches its text."""
    checks: list[bool] = []
    for c in candidates:
        ok = hashlib.sha256(c.text.encode("utf-8")).hexdigest() == c.text_hash
        ok = ok and (
            not c.section_name or c.section_name == c.heading_path[-1] if c.heading_path else True
        )
        if vault is not None:
            ok = ok and (vault / c.path).is_file()
        checks.append(bool(ok))
    return checks


RetrieveFn = Callable[[Question], RetrievalResult]


def variant_retrieve(retriever: Retriever, variant: str, *, top_k: int) -> RetrieveFn:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    mode = "hybrid" if variant.startswith("hybrid") else variant
    rerank = variant == "hybrid+rerank"

    def run(question: Question) -> RetrievalResult:
        return retriever.retrieve(
            question.query,
            profile=question.profile,
            mode=mode,
            top_k=top_k,
            rerank=rerank,
        )

    return run


def evaluate(
    golden: GoldenSet,
    retrieve: RetrieveFn,
    *,
    vault: Path | None = None,
    top_k: int = 10,
) -> list[QuestionOutcome]:
    outcomes: list[QuestionOutcome] = []
    for question in golden.questions:
        try:
            result = retrieve(question)
        except Exception as exc:  # record, never abort the run
            outcomes.append(
                QuestionOutcome(
                    id=question.id,
                    category=question.category,
                    split=question.split,
                    mode=question.mode,
                    profile=question.profile,
                    query=question.query,
                    intent="",
                    retrieved=[],
                    relevant_paths=sorted(question.relevant_paths),
                    hit_ranks=[],
                    first_hit_rank=None,
                    hit_at={},
                    recall_at={},
                    reciprocal_rank=0.0,
                    ndcg_at_10=0.0,
                    authority_top1_match=None,
                    authority_any_top3=None,
                    duplicate_concentration=0.0,
                    citation_ok_fraction=0.0,
                    conflicts=[],
                    elapsed_ms=0.0,
                    features={},
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        checks = _citation_checks(vault, result.candidates)
        outcomes.append(score_question(question, result, citation_ok=checks, top_n=top_k))
    return outcomes


def run_variant(
    golden: GoldenSet,
    *,
    variant: str,
    split: str | None,
    retriever: Retriever,
    conn: sqlite3.Connection,
    settings_snapshot: dict[str, Any],
    vault: Path | None,
    top_k: int = 10,
    include_outcomes: bool = True,
) -> RunRecord:
    subset = golden.subset(split=split) if split else golden
    started = datetime.now(UTC)
    t0 = time.perf_counter()
    outcomes = evaluate(
        subset, variant_retrieve(retriever, variant, top_k=top_k), vault=vault, top_k=top_k
    )
    _ = time.perf_counter() - t0
    overall: Aggregate = aggregate(outcomes)
    snapshot = projection_snapshot(conn)
    finished = datetime.now(UTC)
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sqlite": sqlite3.sqlite_version,
    }
    for pkg in ("fastembed", "sqlite-vec", "numpy"):
        try:
            env[pkg] = pkg_version(pkg)
        except Exception:
            env[pkg] = "unknown"
    return RunRecord(
        variant=variant,
        split=split or "all",
        golden_set=str(golden.source_path) if golden.source_path else "",
        golden_version=golden.version,
        question_count=len(subset.questions),
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        git_sha=git_sha(),
        corpus_fingerprint=snapshot["fingerprint"],
        document_count=snapshot["counts"]["documents"],
        chunk_count=snapshot["counts"]["chunks"],
        vector_count=snapshot["counts"]["vectors"],
        fts_count=snapshot["counts"]["fts"],
        projection_meta=snapshot["meta"],
        settings={**settings_snapshot, "top_k": top_k},
        environment=env,
        peak_rss_mb=_peak_rss_mb(),
        overall=overall.to_dict(),
        by_category={k: v.to_dict() for k, v in per_category(outcomes).items()},
        outcomes=[o.to_dict() for o in outcomes] if include_outcomes else [],
    )


def write_run(record: RunRecord, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = record.started_at.replace(":", "").replace("-", "").split(".")[0].replace("+0000", "Z")
    name = f"{stamp}-{record.variant.replace('+', '-')}-{record.split}.json"
    path = out_dir / name
    path.write_text(
        json.dumps(record.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def load_run(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


_TABLE_METRICS: tuple[tuple[str, str], ...] = (
    ("hit@1", "hit_at.1"),
    ("hit@5", "hit_at.5"),
    ("recall@10", "recall_at.10"),
    ("MRR", "mrr"),
    ("nDCG@10", "ndcg_at_10"),
    ("authority@1", "authority_accuracy_top1"),
    ("dup", "duplicate_concentration"),
    ("cite", "citation_fidelity"),
    ("p50ms", "latency_p50_ms"),
    ("p95ms", "latency_p95_ms"),
)


def _dig(data: dict[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def format_comparison(runs: Sequence[dict[str, Any]], *, category: str | None = None) -> str:
    """Markdown table comparing run records on the same split."""
    header = ["variant", "split", "n"] + [name for name, _ in _TABLE_METRICS] + ["rss_mb"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for run in runs:
        block = run["overall"] if category is None else run["by_category"].get(category, {})
        cells = [str(run.get("variant")), str(run.get("split")), str(block.get("n", ""))]
        for _, key in _TABLE_METRICS:
            value = _dig(block, key)
            if isinstance(value, float):
                cells.append(f"{value:.3f}" if "ms" not in key else f"{value:.0f}")
            else:
                cells.append("-" if value is None else str(value))
        cells.append(str(run.get("peak_rss_mb", "")))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


__all__ = [
    "VARIANTS",
    "RunRecord",
    "evaluate",
    "format_benchmark_markdown",
    "format_comparison",
    "git_sha",
    "load_run",
    "projection_snapshot",
    "regression_report",
    "run_variant",
    "variant_retrieve",
    "write_run",
]


REGRESSION_RULE = {"hit_at_5": 0.02, "mrr": 0.02, "authority_accuracy_top1": 0.05}


def regression_report(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Apply the predeclared regression rule (docs/evaluation.md) to two run records."""
    problems: list[str] = []
    comparable = True
    for key in ("split", "golden_version", "corpus_fingerprint"):
        if baseline.get(key) != candidate.get(key):
            comparable = False
            problems.append(f"{key} differs ({baseline.get(key)!r} vs {candidate.get(key)!r})")
    b = baseline.get("overall", {})
    c = candidate.get("overall", {})
    deltas: dict[str, float | None] = {}
    regression = False
    checks = (
        ("hit_at_5", _dig(b, "hit_at.5"), _dig(c, "hit_at.5")),
        ("mrr", b.get("mrr"), c.get("mrr")),
        (
            "authority_accuracy_top1",
            b.get("authority_accuracy_top1"),
            c.get("authority_accuracy_top1"),
        ),
        ("ndcg_at_10", b.get("ndcg_at_10"), c.get("ndcg_at_10")),
        ("recall_at_10", _dig(b, "recall_at.10"), _dig(c, "recall_at.10")),
        ("latency_p95_ms", b.get("latency_p95_ms"), c.get("latency_p95_ms")),
    )
    for name, before, after in checks:
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            delta = float(after) - float(before)
            deltas[name] = round(delta, 4)
            tolerance = REGRESSION_RULE.get(name)
            if tolerance is not None and delta < -tolerance:
                regression = True
                problems.append(f"{name} dropped by {-delta:.3f} (> {tolerance})")
        else:
            deltas[name] = None
    return {
        "comparable": comparable,
        "regression": regression if comparable else None,
        "deltas": deltas,
        "baseline": {"variant": baseline.get("variant"), "git_sha": baseline.get("git_sha")},
        "candidate": {"variant": candidate.get("variant"), "git_sha": candidate.get("git_sha")},
        "problems": problems,
    }


def format_benchmark_markdown(runs: Sequence[dict[str, Any]]) -> str:
    """Render a benchmarks page from run records: latest run per (golden, split, variant)."""
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for run in runs:
        key = (str(run.get("golden_version")), str(run.get("split")), str(run.get("variant")))
        if key not in latest or str(run.get("started_at")) > str(latest[key].get("started_at")):
            latest[key] = run
    lines = [
        "# Benchmarks",
        "",
        "Generated by `llmwiki eval report` from `evals/runs/`. Latest run per golden set, split and variant.",
        "",
    ]
    for golden in sorted({k[0] for k in latest}):
        for split in ("heldout", "dev", "all"):
            rows = [latest[k] for k in sorted(latest) if k[0] == golden and k[1] == split]
            if not rows:
                continue
            first = rows[0]
            lines.append(f"## Golden {golden} — {split} ({first.get('question_count')} questions)")
            lines.append("")
            lines.append(
                f"Corpus fingerprint `{str(first.get('corpus_fingerprint'))[:12]}`, "
                f"{first.get('document_count')} documents, {first.get('chunk_count')} chunks, "
                f"model `{first.get('projection_meta', {}).get('recipe.embedding_model', '?')}`."
            )
            lines.append("")
            lines.append(format_comparison(rows))
            lines.append("")
            lines.append("Per category (hit@5 / MRR):")
            lines.append("")
            cats = sorted({c for r in rows for c in r.get("by_category", {})})
            header = "| category | " + " | ".join(str(r.get("variant")) for r in rows) + " |"
            lines.append(header)
            lines.append("|" + "---|" * (len(rows) + 1))
            for cat in cats:
                cells = []
                for r in rows:
                    block = r.get("by_category", {}).get(cat, {})
                    hit = _dig(block, "hit_at.5")
                    mrr = block.get("mrr")
                    cells.append(
                        f"{hit:.2f} / {mrr:.2f}"
                        if isinstance(hit, float) and isinstance(mrr, float)
                        else "-"
                    )
                lines.append(f"| {cat} | " + " | ".join(cells) + " |")
            lines.append("")
    lines.append("Gates and recorded decisions: `docs/evaluation.md`.")
    return "\n".join(lines) + "\n"
