"""The public sample vault and golden set run end to end through the harness."""

from __future__ import annotations

from pathlib import Path

from llmwiki import db as dbmod
from llmwiki.config import Settings
from llmwiki.evaluation.golden import load_golden, stratification_report, validate_golden
from llmwiki.evaluation.metrics import aggregate
from llmwiki.evaluation.runner import evaluate, run_variant, write_run
from llmwiki.indexer import Indexer
from llmwiki.retrieval import Retriever
from tests.helpers import SAMPLE_KEYWORDS, KeywordEmbedder

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_VAULT = ROOT / "evals" / "sample-vault"
SAMPLE_GOLDEN = ROOT / "evals" / "golden" / "sample-vault.json"


def test_sample_golden_validates_against_sample_vault() -> None:
    golden = load_golden(SAMPLE_GOLDEN)
    assert validate_golden(golden, vault=SAMPLE_VAULT) == []
    assert stratification_report(golden, minimum_total=16) == []


def test_sample_set_runs_through_the_harness(tmp_path: Path) -> None:
    settings = Settings(vault_path=SAMPLE_VAULT, db_path=tmp_path / "sample.sqlite")
    embedder = KeywordEmbedder(SAMPLE_KEYWORDS)
    assert not Indexer(settings, embedder=embedder).run().errors
    golden = load_golden(SAMPLE_GOLDEN)
    with dbmod.connect(settings.db_path) as conn:
        retriever = Retriever(conn, embedder=embedder, settings=settings)
        outcomes = evaluate(
            golden,
            lambda q: retriever.retrieve(q.query, profile=q.profile, mode="hybrid", top_k=10),
            vault=SAMPLE_VAULT,
        )
        agg = aggregate(outcomes)
        assert agg.n == 16 and agg.n_abstain == 2 and agg.errors == 0
        assert agg.hit_at["5"] is not None and agg.hit_at["5"] >= 0.8
        record = run_variant(
            golden,
            variant="hybrid",
            split="heldout",
            retriever=retriever,
            conn=conn,
            settings_snapshot={},
            vault=SAMPLE_VAULT,
        )
    path = write_run(record, tmp_path / "runs")
    assert path.exists() and record.question_count == 8
