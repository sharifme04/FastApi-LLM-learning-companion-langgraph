"""Recommender agent — suggests next topic based on learning state."""

from __future__ import annotations

import json
import logging

import redis.asyncio as aioredis

from app.agents.state import LearningState
from app.services.cost import calculate_cost
from app.services.llm_client import call_anthropic

logger = logging.getLogger("learning_companion")


PROMPT = """You are a study advisor. Given the learner's state (topics covered, confidence per topic, recent quiz scores), recommend the next 1-2 topics to study and why.

Return JSON ONLY:
{{"recommendation": "<one paragraph>", "next_topics": ["<topic>", "<topic>"]}}

Learner state:
{state}

Recent message from learner: {msg}
"""


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    return s


async def recommend(
    state: LearningState,
    redis: aioredis.Redis,
) -> LearningState:
    learner_state = state.get("learning_state") or {}
    res = await call_anthropic(
        redis,
        system="Output strict JSON only.",
        user_message=PROMPT.format(state=learner_state, msg=state["user_message"]),
        max_tokens=400,
    )
    raw = _strip_fences(res["text"])
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "recommendation": "Pick the topic with the lowest confidence and review it next.",
            "next_topics": [],
        }
    state["recommendation"] = parsed.get("recommendation", "")
    state["input_tokens"] = state.get("input_tokens", 0) + res["input_tokens"]
    state["output_tokens"] = state.get("output_tokens", 0) + res["output_tokens"]
    state["cost"] = calculate_cost(state["input_tokens"], state["output_tokens"])
    state["model_used"] = res["model"]
    state["final_message"] = parsed.get("recommendation", "")
    state["final_sources"] = []

    # Persist recommended topics
    learner_state["recommended_next"] = parsed.get("next_topics", [])
    state["learning_state"] = learner_state

    logger.info("Recommendation produced", extra={"session": state.get("session_id")})
    return state
