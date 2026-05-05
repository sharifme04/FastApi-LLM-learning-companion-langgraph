"""LangGraph state machine for the 3-agent learning flow.

  classify ─┬─→ responder    ─→ END
            ├─→ assessor     ─→ END
            └─→ recommender  ─→ END

The classifier is intent-classification only (rule-based + LLM tiebreak).
After classification, the graph routes to exactly one agent.

If LangGraph isn't importable (e.g. minimal test envs), we fall back to a
manual sequential dispatcher with the same interface.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Callable

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.assessor import generate_quiz
from app.agents.recommender import recommend
from app.agents.responder import respond
from app.agents.state import LearningState

logger = logging.getLogger("learning_companion")


_QUIZ_HINTS = re.compile(r"\b(quiz|test me|practice|ask me|assess)\b", re.I)
_RECOMMEND_HINTS = re.compile(r"\b(what (should|next)|recommend|suggest|next topic)\b", re.I)


def classify_intent(state: LearningState) -> LearningState:
    """Cheap rule-based intent classifier — no LLM call."""
    msg = state.get("user_message", "")
    if _QUIZ_HINTS.search(msg):
        state["agent_action"] = "assess"
    elif _RECOMMEND_HINTS.search(msg):
        state["agent_action"] = "recommend"
    else:
        state["agent_action"] = "respond"
    return state


def _route(state: LearningState) -> str:
    return state.get("agent_action", "respond")


# ----------- Build the graph (with LangGraph if available, else fallback) ----------- #

_HAS_LANGGRAPH = False
try:
    from langgraph.graph import END, StateGraph

    _HAS_LANGGRAPH = True
except ImportError:
    StateGraph = None  # type: ignore[assignment]
    END = "END"  # type: ignore[assignment]


def build_graph(db: AsyncSession, redis: aioredis.Redis) -> Callable[[LearningState], Any]:
    """Return an async callable that runs the full state machine on a state dict."""
    if _HAS_LANGGRAPH:
        return _build_langgraph(db, redis)
    return _build_fallback(db, redis)


def _build_langgraph(db: AsyncSession, redis: aioredis.Redis) -> Callable[[LearningState], Any]:
    async def _classify(state: LearningState) -> LearningState:
        return classify_intent(state)

    async def _respond(state: LearningState) -> LearningState:
        return await respond(state, db, redis)

    async def _assess(state: LearningState) -> LearningState:
        return await generate_quiz(state, redis)

    async def _recommend(state: LearningState) -> LearningState:
        return await recommend(state, redis)

    g = StateGraph(LearningState)
    g.add_node("classify", _classify)
    g.add_node("respond", _respond)
    g.add_node("assess", _assess)
    g.add_node("recommend", _recommend)

    g.set_entry_point("classify")
    g.add_conditional_edges(
        "classify",
        _route,
        {"respond": "respond", "assess": "assess", "recommend": "recommend"},
    )
    g.add_edge("respond", END)
    g.add_edge("assess", END)
    g.add_edge("recommend", END)

    compiled = g.compile()

    async def runner(state: LearningState) -> LearningState:
        if "trace_id" not in state:
            state["trace_id"] = uuid.uuid4().hex[:12]
        return await compiled.ainvoke(state)

    return runner


def _build_fallback(db: AsyncSession, redis: aioredis.Redis) -> Callable[[LearningState], Any]:
    """Minimal sequential equivalent if langgraph isn't installed."""
    async def runner(state: LearningState) -> LearningState:
        if "trace_id" not in state:
            state["trace_id"] = uuid.uuid4().hex[:12]
        state = classify_intent(state)
        action = state.get("agent_action", "respond")
        if action == "assess":
            return await generate_quiz(state, redis)
        if action == "recommend":
            return await recommend(state, redis)
        return await respond(state, db, redis)

    return runner
