"""Responder agent — RAG over the user's learning materials."""

from __future__ import annotations

import logging
from typing import Any

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.state import LearningState
from app.services.cost import calculate_cost
from app.services.llm_client import call_anthropic
from app.services.retriever import retrieve

logger = logging.getLogger("learning_companion")


SYSTEM_PROMPT = """You are a helpful, patient learning assistant. Answer the user's question using only the provided context passages from their learning materials. Cite passages by [n] inline. If the context doesn't contain the answer, say so explicitly.

Be brief and pedagogical — if the user is at an early confidence level on this topic (provided in 'Learner state'), explain foundational concepts; if confidence is high, skip the basics.

Return plain prose. Do not output JSON."""


async def respond(
    state: LearningState,
    db: AsyncSession,
    redis: aioredis.Redis,
) -> LearningState:
    """Look up relevant material via RAG, then ask Claude to answer."""
    question = state["user_message"]
    learner_state = state.get("learning_state") or {}

    retrieved = await retrieve(db, question)
    state["retrieved"] = [
        {
            "chunk_id": r.chunk_id,
            "document_filename": r.document_filename,
            "page_start": r.page_start,
            "page_end": r.page_end,
            "score": round(r.relevance_score, 4),
            "text_preview": r.text[:300],
        }
        for r in retrieved
    ]

    if retrieved:
        ctx = "\n".join(
            f"[{i + 1}] from {r.document_filename}: {r.text}"
            for i, r in enumerate(retrieved)
        )
    else:
        ctx = "(no relevant passages found in user's materials)"

    user_msg = (
        f"Question: {question}\n\n"
        f"Learner state: {learner_state}\n\n"
        f"Context passages:\n{ctx}"
    )

    res = await call_anthropic(redis, SYSTEM_PROMPT, user_msg, max_tokens=600)
    state["response_text"] = res["text"]
    state["input_tokens"] = state.get("input_tokens", 0) + res["input_tokens"]
    state["output_tokens"] = state.get("output_tokens", 0) + res["output_tokens"]
    state["cost"] = calculate_cost(state["input_tokens"], state["output_tokens"])
    state["model_used"] = res["model"]
    state["final_message"] = res["text"]
    state["final_sources"] = state["retrieved"]

    logger.info(
        "Responder produced answer",
        extra={
            "session": state.get("session_id"),
            "n_passages": len(retrieved),
            "model": res["model"],
        },
    )
    return state
