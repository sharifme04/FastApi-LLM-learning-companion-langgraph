"""Embedding + reranking via sentence-transformers.

Singleton models loaded lazily on first use. Both the bi-encoder
(all-MiniLM-L6-v2 by default) and the cross-encoder are kept in memory
for the lifetime of the process.

For tests, the singletons can be overridden by `set_embedder()` /
`set_reranker()` to inject deterministic fakes.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol

import numpy as np

from app.config import get_settings

logger = logging.getLogger("learning_companion")
settings = get_settings()


class EmbedderProtocol(Protocol):
    def encode(self, texts: list[str], batch_size: int = ...) -> np.ndarray: ...


class RerankerProtocol(Protocol):
    def predict(self, pairs: list[list[str]]) -> np.ndarray: ...


_embedder: EmbedderProtocol | None = None
_reranker: RerankerProtocol | None = None
_lock = threading.Lock()


class _RealEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer(model_name)

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        return self._m.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


class _RealReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._m = CrossEncoder(model_name)

    def predict(self, pairs: list[list[str]]) -> np.ndarray:
        return self._m.predict(pairs)


def get_embedder() -> EmbedderProtocol:
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                logger.info("Loading embedder", extra={"model": settings.embedding_model})
                _embedder = _RealEmbedder(settings.embedding_model)
    return _embedder


def get_reranker() -> RerankerProtocol:
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                logger.info("Loading reranker", extra={"model": settings.reranker_model})
                _reranker = _RealReranker(settings.reranker_model)
    return _reranker


def set_embedder(e: EmbedderProtocol | None) -> None:
    """Test hook — inject a deterministic fake."""
    global _embedder
    _embedder = e


def set_reranker(r: RerankerProtocol | None) -> None:
    global _reranker
    _reranker = r


# ---------------- Convenience helpers ---------------- #


def embed_texts(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """Embed a list of strings, returning a list of float lists (JSON-friendly)."""
    if not texts:
        return []
    emb = get_embedder().encode(texts, batch_size=batch_size)
    if isinstance(emb, np.ndarray):
        return emb.tolist()
    return [list(map(float, row)) for row in emb]


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]


def rerank(query: str, candidates: list[str]) -> list[float]:
    """Score (query, candidate) pairs. Higher score = more relevant."""
    if not candidates:
        return []
    pairs = [[query, c] for c in candidates]
    scores = get_reranker().predict(pairs)
    if isinstance(scores, np.ndarray):
        return scores.tolist()
    return [float(s) for s in scores]
