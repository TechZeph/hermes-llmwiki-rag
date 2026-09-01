"""Embedding backends (Phase 3).

The :class:`Embedder` ABC is the contract every backend satisfies. The
:class:`FastEmbedEmbedder` is the only real implementation we ship in
Phase 3 — it's local-only, deterministic, and matches the
``chunk_embeddings.embedding`` column dimension in the schema.

Design notes (locked at Phase 3 design time, see wiki/projects/
hermes-llmwiki-rag/plan.md and the 2026-08-28 discussion):

- The interface is a class, not a function. We hold the model handle
  across calls so the first-embed cost (model load) happens once per
  process.
- Output is a plain Python ``list[float]`` per input text. We do not
  return numpy arrays because the vector store layer is the only
  consumer and it converts once at the boundary; passing numpy up
  the stack would force numpy typing everywhere with no payoff.
- Batch API. Even single-text callers go through ``embed()``; we
  accept a sequence so FastEmbed's optimised batch path is always
  used.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from importlib.metadata import version


def model_provenance(model_name: str) -> dict[str, str]:
    """Return the local FastEmbed package and registered artifact identity.

    FastEmbed's registry exposes a package/version and model artifact source,
    but not an immutable artifact checksum. Operators must therefore retain the
    provisioned cache when exact byte-for-byte reproduction matters.
    """
    from fastembed import TextEmbedding

    artifact_source = "unknown"
    for model in TextEmbedding.list_supported_models():
        if str(model.get("model")) != model_name:
            continue
        sources = model.get("sources", {})
        if isinstance(sources, dict):
            source = sources.get("hf") or sources.get("url")
            if isinstance(source, str) and source:
                artifact_source = source
        break
    return {
        "embedding.backend": "fastembed",
        "embedding.backend_version": version("fastembed"),
        "embedding.artifact_source": artifact_source,
    }


class Embedder(ABC):
    """Abstract embedding backend.

    Implementations: :class:`FastEmbedEmbedder` (Phase 3). Future
    work may add an Ollama backend or an OpenAI-compatible backend for
    benchmarking.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier (e.g. ``BAAI/bge-small-en-v1.5``)."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the embedding vectors this embedder produces."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input.

        The order of the returned list matches the order of the input.
        Empty input returns an empty list.
        """


class FastEmbedEmbedder(Embedder):
    """Local-only embedder wrapping FastEmbed's ``TextEmbedding``.

    FastEmbed handles model download (Hugging Face) and caching. We
    expose ``FASTEMBED_CACHE_PATH`` for the standard env override.

    Parameters
    ----------
    model_name:
        Any model supported by FastEmbed. Defaults to
        ``BAAI/bge-small-en-v1.5`` (the plan's recommended default,
        384-dim, English, ~50 MB).
    cache_path:
        Optional override for FastEmbed's model cache. ``None`` means
        use FastEmbed's default (typically ``~/.cache/fastembed``).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        cache_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._model_name = model_name
        self._dim: int | None = None
        if cache_path is not None:
            os.environ.setdefault("FASTEMBED_CACHE_PATH", str(cache_path))
        # Lazy import: fastembed is heavy and the import-time model
        # list fetch is not needed for every CLI invocation.
        from fastembed import TextEmbedding

        self._impl = TextEmbedding(model_name=model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        # FastEmbed does not expose ``dim`` until the model has been
        # run at least once. Probe with a tiny string on first access
        # and cache. The model is already loaded by ``__init__`` so
        # this is a forward pass, not a download.
        if self._dim is None:
            probe = self.embed(["dim-probe"])
            self._dim = len(probe[0])
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        # FastEmbed's ``embed`` is a generator of numpy arrays.
        import numpy as np  # local import to keep module-load light

        vectors: list[list[float]] = []
        for arr in self._impl.embed(texts):
            assert isinstance(arr, np.ndarray), f"expected ndarray, got {type(arr)}"
            vectors.append(arr.astype(float).tolist())
        return vectors


__all__ = ["Embedder", "FastEmbedEmbedder"]
