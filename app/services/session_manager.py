"""Session orchestration: load state → run graph → persist message + state.

Also flips the Redis cost-fallback flag when the daily total crosses
settings.cost_fallback_threshold, so the next call uses Haiku.
"""

from __future__ import annotations

import logging
from typing import Any

import redis.asyncio as aioredis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import build_graph
from app.agents.state import LearningState
from app.config import get_settings
from app.models.message import Message
from app.models.session import Session
from app.services.cost import daily_cost_total
from app.services.llm_client import set_fallback_flag

logger = logging.getLogger("learning_companion")
settings = get_settings()


async def get_or_create_session(
    db: AsyncSession, session_id: int | None, user_id: str = "anonymous", title: str = "New session"
) -> Session:
    if session_id is not None:
        sess = (
            await db.execute(select(Session).where(Session.id == session_id))
        ).scalar_one_or_none()
        if sess is not None:
            return sess
    sess = Session(user_id=user_id, title=title, learning_state={"topics": []})
    db.add(sess)
    await db.flush()
    return sess


async def chat_turn(
    db: AsyncSession,
    redis: aioredis.Redis,
    session: Session,
    user_message: str,
) -> Message:
    """Run one chat turn through the LangGraph state machine, persist results."""
    # Persist user message first
    user_row = Message(
        session_id=session.id,
        role="user",
        content=user_message,
    )
    db.add(user_row)
    await db.flush()

    learning_state = session.learning_state or {"topics": []}

    state: LearningState = {
        "session_id": session.id,
        "user_message": user_message,
        "user_id": session.user_id,
        "learning_state": dict(learning_state),
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": 0.0,
        "model_used": "",
        "cache_hit": False,
        "retrieved": [],
        "final_message": "",
        "final_sources": [],
    }

    runner = build_graph(db, redis)
    final = await runner(state)

    assistant = Message(
        session_id=session.id,
        role="assistant",
        content=final.get("final_message", "") or "(no response)",
        agent_action=final.get("agent_action"),
        sources=final.get("final_sources") or [],
        input_tokens=final.get("input_tokens", 0),
        output_tokens=final.get("output_tokens", 0),
        cost=final.get("cost", 0.0),
        cache_hit=final.get("cache_hit", False),
        model_used=final.get("model_used"),
        trace_id=final.get("trace_id"),
    )
    db.add(assistant)

    # Update session learning_state if the agents touched it
    new_state = final.get("learning_state")
    if isinstance(new_state, dict):
        session.learning_state = new_state

    await db.flush()

    # Cost-driven fallback flag
    try:
        total_today = await daily_cost_total(db)
        await set_fallback_flag(redis, total_today >= settings.cost_fallback_threshold)
    except Exception as e:
        logger.warning("Could not update cost flag: %s", e)

    logger.info(
        "Chat turn complete",
        extra={
            "session": session.id,
            "agent": final.get("agent_action"),
            "cost": final.get("cost"),
            "model": final.get("model_used"),
        },
    )
    return assistant
