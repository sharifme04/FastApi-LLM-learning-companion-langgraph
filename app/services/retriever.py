"""Retrieval pipeline: vector search + cross-encoder reranking.

On Postgres we use pgvector cosine distance (`<=>`) for the ANN-style top-k.
On SQLite (test fallback) we load all chunks and compute cosine similarity
in Python — fine for small fixtures.

After the top-K_VECTOR candidates are retrieved, a cross-encoder reranks
the list and we return the top-K_RERANK with their relevance scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.chunk import Chunk
from app.models.document import Document
from app.services.embedder import embed_query, rerank

logger = logging.getLogger("learning_companion")
settings = get_settings()


@dataclass(slots=True)
class Retrieved:
    chunk_id: int
    chunk_index: int
    text: str
    document_id: int
    document_filename: str
    page_start: Optional[int]
    page_end: Optional[int]
    relevance_score: float


def _cosine(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0.0:
        return 0.0
    return float(np.dot(av, bv) / denom)


async def vector_search(
    db: AsyncSession,
    query_embedding: list[float],
    top_k: int,
    document_ids: Optional[list[int]] = None,
) -> list[tuple[Chunk, Document, float]]:
    """Return top-k chunks (Chunk, Document, similarity) ordered by similarity."""
    dialect = db.bind.dialect.name if db.bind else "sqlite"

    if dialect == "postgresql":
        # Use pgvector cosine distance: a <=> b → 0 means identical, 2 means opposite.
        # Similarity = 1 - distance.
        from sqlalchemy import bindparam, literal_column

        # Format the embedding as pgvector literal
        emb_literal = "[" + ",".join(repr(float(x)) for x in query_embedding) + "]"
        sql = f"""
            SELECT c.id, c.document_id, c.chunk_index, c.text, c.page_start, c.page_end,
                   d.filename,
                   1 - (c.embedding <=> '{emb_literal}'::vector) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
        """
        if document_ids:
            ids = ",".join(str(int(x)) for x in document_ids)
            sql += f" AND c.document_id IN ({ids})"
        sql += f" ORDER BY c.embedding <=> '{emb_literal}'::vector ASC LIMIT {int(top_k)}"

        from sqlalchemy import text as sql_text

        result = await db.execute(sql_text(sql))
        rows = result.all()

        # We need ORM objects for chunk; build them by id.
        out: list[tuple[Chunk, Document, float]] = []
        for r in rows:
            chunk = (
                await db.execute(select(Chunk).where(Chunk.id == r.id))
            ).scalar_one()
            doc = (
                await db.execute(select(Document).where(Document.id == r.document_id))
            ).scalar_one()
            out.append((chunk, doc, float(r.similarity)))
        return out

    # ---- SQLite fallback: pull all chunks and compute cosine in Python ----
    stmt = select(Chunk, Document).join(Document, Chunk.document_id == Document.id)
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))
    rows = (await db.execute(stmt)).all()

    scored: list[tuple[Chunk, Document, float]] = []
    for chunk, doc in rows:
        if chunk.embedding is None:
            continue
        sim = _cosine(query_embedding, chunk.embedding)
        scored.append((chunk, doc, sim))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]


async def retrieve(
    db: AsyncSession,
    question: str,
    top_k_vector: Optional[int] = None,
    top_k_rerank: Optional[int] = None,
    document_ids: Optional[list[int]] = None,
) -> list[Retrieved]:
    """End-to-end retrieval: embed → vector search → cross-encoder rerank."""
    kv = top_k_vector or settings.top_k_vector
    kr = top_k_rerank or settings.top_k_rerank

    q_emb = embed_query(question)
    candidates = await vector_search(db, q_emb, top_k=kv, document_ids=document_ids)
    if not candidates:
        return []

    texts = [c.text for c, _, _ in candidates]
    rerank_scores = rerank(question, texts)

    enriched = list(zip(candidates, rerank_scores))
    enriched.sort(key=lambda pair: pair[1], reverse=True)

    out: list[Retrieved] = []
    for (chunk, doc, _vec_score), rscore in enriched[:kr]:
        out.append(
            Retrieved(
                chunk_id=chunk.id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                document_id=doc.id,
                document_filename=doc.filename,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                relevance_score=float(rscore),
            )
        )

    logger.info(
        "Retrieved chunks",
        extra={
            "question_len": len(question),
            "candidates": len(candidates),
            "returned": len(out),
        },
    )
    return out
