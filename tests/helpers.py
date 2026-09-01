"""Shared test helpers: deterministic embedders and synthetic vault builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from llmwiki.embeddings import Embedder

PROD_DIM = 384


class KeywordEmbedder(Embedder):
    """Toy embedder: each known keyword owns one dimension.

    A text containing keyword ``K[i]`` gets ``1.0`` at index ``i``. A query
    that contains exactly one keyword therefore has cosine distance 0 to
    every chunk mentioning that keyword, which makes dense assertions exact.
    """

    def __init__(self, keywords: Sequence[str], dim: int = PROD_DIM) -> None:
        if len(keywords) > dim:
            raise ValueError("too many keywords for the embedding dimension")
        self._keywords = tuple(keywords)
        self._index = {kw: i for i, kw in enumerate(self._keywords)}
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "keyword-embedder"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            lower = text.lower()
            for kw, idx in self._index.items():
                if kw in lower:
                    vec[idx] = 1.0
            out.append(vec)
        return out


def write_vault(root: Path, files: Mapping[str, str]) -> Path:
    """Create a vault directory with the given ``{relative_path: content}`` files."""
    root.mkdir(parents=True, exist_ok=True)
    (root / ".obsidian").mkdir(exist_ok=True)
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


SAMPLE_VAULT: dict[str, str] = {
    "wiki/sqlite-vec.md": (
        "# sqlite-vec\n\n**Summary**: Vector search extension for SQLite.\n\n---\n\n"
        "## Storage\n\nsqlite-vec stores float32 vectors in a vec0 virtual table. "
        "The `chunk_embeddings` table uses float[384].\n\n"
        "## Limits\n\nKNN queries reject k values above 4096 with an error.\n"
    ),
    "wiki/fastembed.md": (
        "# FastEmbed\n\n**Summary**: Local ONNX embedding runtime.\n\n---\n\n"
        "## Models\n\nThe default model is BAAI/bge-small-en-v1.5 which is 384 dimensional.\n\n"
        "## Memory\n\nThe ONNX arena grows with the largest batch ever embedded.\n"
    ),
    "wiki/index-tools.md": "# Tools index\n\n- [[sqlite-vec]] vector storage\n- [[fastembed]] embeddings\n",
    "wiki/log.md": "# Log\n\n## [2026-08-30] index | sqlite-vec landed\n\nAdded sqlite-vec storage for embeddings.\n",
    "wiki/projects/rag/current-state.md": (
        "# RAG current state\n\n## Status\n\nThe sqlite-vec projection currently holds 5000 vectors "
        "and the embeddings pipeline is stable.\n"
    ),
    "wiki/projects/rag/decisions.md": (
        "# RAG decisions\n\n## [2026-08-30] storage\n\nWe chose sqlite-vec over faiss because "
        "it keeps embeddings inside SQLite.\n"
    ),
    "wiki/projects/rag/index.md": "# RAG index\n\n- [[current-state]]\n- [[decisions]]\n",
    "raw/papers/rrf-paper.md": (
        "# RRF paper\n\nReciprocal rank fusion outperforms Condorcet and individual rank learning methods. "
        "The paper evaluates sqlite-vec? No, it evaluates fusion of ranked lists.\n"
    ),
    "Clippings/ideas/vector-idea.md": "# Idea\n\nMaybe sqlite-vec could store embeddings for images too.\n",
    "TODO.md": "- [ ] embeddings cleanup\n",
}

SAMPLE_KEYWORDS = ("sqlite-vec", "fastembed", "embeddings", "fusion", "arena", "faiss")
