"""LangGraph state schema.

This is a TypedDict (LangGraph requirement). Each agent node mutates
specific fields. The graph routes based on `agent_action` after the
classifier node.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class LearningState(TypedDict, total=False):
    # Session context
    session_id: int
    user_message: str
    user_id: str

    # Persistent learning state (loaded from DB at start, written back at end)
    learning_state: dict[str, Any]

    # Routing
    agent_action: str  # "respond" | "assess" | "recommend"

    # Responder fields
    retrieved: list[dict[str, Any]]
    response_text: str

    # Assessor fields
    quiz_question: Optional[str]
    quiz_topic: Optional[str]
    quiz_rubric: Optional[str]
    quiz_user_answer: Optional[str]
    quiz_score: Optional[float]
    quiz_feedback: Optional[str]

    # Recommender fields
    recommendation: Optional[str]

    # Cost / model
    input_tokens: int
    output_tokens: int
    cost: float
    model_used: str
    trace_id: str
    cache_hit: bool

    # Output assembly
    final_message: str
    final_sources: list[dict[str, Any]]
