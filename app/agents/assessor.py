"""Assessor agent — generates quiz questions, grades user answers.

Two modes:
- Generate: produce a quiz question + rubric for the topic the user just
  asked about (or any uncovered topic).
- Grade: given a question + rubric + user answer, return a 0..1 score and
  feedback. Update confidence in learning_state.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import redis.asyncio as aioredis

from app.agents.state import LearningState
from app.config import get_settings
from app.services.cost import calculate_cost
from app.services.llm_client import call_anthropic

logger = logging.getLogger("learning_companion")
settings = get_settings()


GEN_PROMPT = """You are a learning assessor. Generate ONE concise quiz question for the topic given. Include a brief grading rubric (1-3 bullet points listing what a correct answer should mention).

Return JSON ONLY:
{{"topic": "<topic>", "question": "<question>", "rubric": "<rubric>"}}

Topic: {topic}
Learner state (for difficulty calibration): {learner_state}
"""


GRADE_PROMPT = """You are a strict but kind grader. Given a question, a rubric, and the user's answer, score 0.0..1.0 and write 1-2 sentences of feedback.

Return JSON ONLY:
{{"score": 0.0..1.0, "feedback": "<short string>"}}

Question: {question}
Rubric: {rubric}
User answer: {user_answer}
"""


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    return s


def _safe_json(raw: str, fallback: dict) -> dict:
    try:
        return json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return fallback


def _topic_from_message(msg: str) -> str:
    """Heuristic: pick the first noun-phrase-ish token as the topic."""
    words = [w for w in re.findall(r"\w+", msg) if len(w) > 3]
    return words[0].title() if words else "general"


async def generate_quiz(
    state: LearningState,
    redis: aioredis.Redis,
) -> LearningState:
    user_msg = state["user_message"]
    topic = _topic_from_message(user_msg)
    state["quiz_topic"] = topic
    learner_state = state.get("learning_state") or {}

    res = await call_anthropic(
        redis,
        system="Output strict JSON only.",
        user_message=GEN_PROMPT.format(topic=topic, learner_state=learner_state),
        max_tokens=300,
    )
    parsed = _safe_json(
        res["text"],
        fallback={
            "topic": topic,
            "question": f"Briefly explain the key concept of {topic}.",
            "rubric": f"A correct answer should explain the core idea of {topic} concisely.",
        },
    )
    state["quiz_question"] = parsed.get("question", "")
    state["quiz_rubric"] = parsed.get("rubric", "")
    state["input_tokens"] = state.get("input_tokens", 0) + res["input_tokens"]
    state["output_tokens"] = state.get("output_tokens", 0) + res["output_tokens"]
    state["cost"] = calculate_cost(state["input_tokens"], state["output_tokens"])
    state["model_used"] = res["model"]

    state["final_message"] = (
        f"📝 Quiz on **{parsed.get('topic', topic)}**:\n\n{parsed.get('question', '')}"
    )
    state["final_sources"] = []
    logger.info("Quiz generated", extra={"topic": topic})
    return state


async def grade_quiz(
    state: LearningState,
    redis: aioredis.Redis,
    question: str,
    rubric: str,
    user_answer: str,
    topic: str | None = None,
) -> tuple[float, str, dict]:
    """Returns (score, feedback, llm_meta)."""
    res = await call_anthropic(
        redis,
        system="Output strict JSON only.",
        user_message=GRADE_PROMPT.format(question=question, rubric=rubric, user_answer=user_answer),
        max_tokens=300,
    )
    parsed = _safe_json(
        res["text"],
        fallback={"score": 0.5, "feedback": "Could not parse grader output."},
    )
    score = float(parsed.get("score", 0.5))
    score = max(0.0, min(1.0, score))
    feedback = str(parsed.get("feedback", ""))

    return (
        score,
        feedback,
        {
            "input_tokens": res["input_tokens"],
            "output_tokens": res["output_tokens"],
            "model": res["model"],
            "topic": topic,
        },
    )


def update_confidence(
    learning_state: dict[str, Any],
    topic: str,
    new_score: float,
    alpha: float | None = None,
    beta: float | None = None,
) -> dict[str, Any]:
    """new_confidence = α · old + β · new_score (bounded 0..1)."""
    a = alpha if alpha is not None else settings.confidence_alpha
    b = beta if beta is not None else settings.confidence_beta
    topics = learning_state.setdefault("topics", [])
    for t in topics:
        if t.get("name", "").lower() == topic.lower():
            old = float(t.get("confidence", 0.5))
            t["confidence"] = round(max(0.0, min(1.0, a * old + b * new_score)), 3)
            t["last_quiz_score"] = round(new_score, 3)
            return learning_state
    topics.append(
        {"name": topic, "confidence": round(b * new_score + a * 0.5, 3),
         "last_quiz_score": round(new_score, 3)}
    )
    return learning_state
