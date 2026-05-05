"""Schemas for sessions, chat messages, quizzes."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    user_id: str = "anonymous"
    title: str = "New session"


class SessionSummary(BaseModel):
    id: int
    user_id: str
    title: str
    learning_state: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    message_id: int
    session_id: int
    role: str
    content: str
    agent_action: Optional[str] = None
    sources: Optional[list[dict[str, Any]]] = None
    cost: float
    cache_hit: bool
    model_used: Optional[str] = None
    trace_id: Optional[str] = None
    learning_state: Optional[dict[str, Any]] = None
    created_at: datetime


class QuizQuestionResponse(BaseModel):
    id: int
    topic: Optional[str] = None
    question: str
    grading_rubric: Optional[str] = None
    score: Optional[float] = None
    feedback: Optional[str] = None
    answered_at: Optional[datetime] = None


class QuizAnswerRequest(BaseModel):
    answer: str = Field(..., min_length=1)


class ProgressResponse(BaseModel):
    session_id: int
    learning_state: dict[str, Any]
    total_messages: int
    total_quiz_questions: int
    total_cost: float


class EvalReport(BaseModel):
    last_run_at: Optional[datetime] = None
    metrics: dict[str, float]
    threshold_correctness: float
    threshold_clarity: float
    passes_thresholds: bool
    num_samples: int
    notes: Optional[str] = None
