"""Session + chat + quiz + progress endpoints."""

import logging
from datetime import datetime, timezone

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.assessor import grade_quiz, update_confidence
from app.database import get_db
from app.models.message import Message
from app.models.quiz_question import QuizQuestion
from app.models.session import Session
from app.redis_client import get_redis
from app.schemas.sessions import (
    ChatRequest,
    ChatResponse,
    ProgressResponse,
    QuizAnswerRequest,
    QuizQuestionResponse,
    SessionCreate,
    SessionSummary,
)
from app.services.session_manager import chat_turn, get_or_create_session

logger = logging.getLogger("learning_companion")
router = APIRouter(tags=["Sessions"])


@router.post("/sessions", response_model=SessionSummary, status_code=201)
async def create_session(
    payload: SessionCreate,
    db: AsyncSession = Depends(get_db),
) -> SessionSummary:
    sess = await get_or_create_session(db, None, user_id=payload.user_id, title=payload.title)
    return SessionSummary.model_validate(sess, from_attributes=True)


@router.get("/sessions/{session_id}", response_model=SessionSummary)
async def get_session(session_id: int, db: AsyncSession = Depends(get_db)) -> SessionSummary:
    sess = (await db.execute(select(Session).where(Session.id == session_id))).scalar_one_or_none()
    if sess is None:
        raise HTTPException(404, f"Session {session_id} not found")
    return SessionSummary.model_validate(sess, from_attributes=True)


@router.post("/sessions/{session_id}/message", response_model=ChatResponse)
async def post_message(
    session_id: int,
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
) -> ChatResponse:
    sess = (await db.execute(select(Session).where(Session.id == session_id))).scalar_one_or_none()
    if sess is None:
        raise HTTPException(404, f"Session {session_id} not found")

    assistant = await chat_turn(db, redis, sess, payload.content)

    # Persist quiz question if assessor produced one
    if assistant.agent_action == "assess":
        # The assistant's content begins with "📝 Quiz on **<topic>**:"
        topic = None
        question_text = assistant.content
        if "**" in assistant.content:
            try:
                topic = assistant.content.split("**")[1]
                question_text = assistant.content.split(":\n\n", 1)[-1]
            except IndexError:
                pass
        db.add(
            QuizQuestion(
                session_id=session_id,
                topic=topic,
                question=question_text,
                grading_rubric=None,  # held in memory; in this minimal version we re-derive at grading time
            )
        )
        await db.flush()

    return ChatResponse(
        message_id=assistant.id,
        session_id=session_id,
        role=assistant.role,
        content=assistant.content,
        agent_action=assistant.agent_action,
        sources=assistant.sources,
        cost=assistant.cost,
        cache_hit=assistant.cache_hit,
        model_used=assistant.model_used,
        trace_id=assistant.trace_id,
        learning_state=sess.learning_state,
        created_at=assistant.created_at,
    )


@router.get("/sessions/{session_id}/quiz", response_model=list[QuizQuestionResponse])
async def list_quiz(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> list[QuizQuestionResponse]:
    rows = (
        await db.execute(
            select(QuizQuestion)
            .where(QuizQuestion.session_id == session_id)
            .order_by(QuizQuestion.created_at.desc())
        )
    ).scalars().all()
    return [
        QuizQuestionResponse(
            id=q.id,
            topic=q.topic,
            question=q.question,
            grading_rubric=q.grading_rubric,
            score=q.score,
            feedback=q.feedback,
            answered_at=q.answered_at,
        )
        for q in rows
    ]


@router.post(
    "/sessions/{session_id}/quiz/{question_id}/answer", response_model=QuizQuestionResponse
)
async def answer_quiz(
    session_id: int,
    question_id: int,
    payload: QuizAnswerRequest,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
) -> QuizQuestionResponse:
    sess = (await db.execute(select(Session).where(Session.id == session_id))).scalar_one_or_none()
    if sess is None:
        raise HTTPException(404, "Session not found")
    qq = (
        await db.execute(
            select(QuizQuestion).where(
                QuizQuestion.id == question_id, QuizQuestion.session_id == session_id
            )
        )
    ).scalar_one_or_none()
    if qq is None:
        raise HTTPException(404, "Quiz question not found")

    rubric = qq.grading_rubric or "A correct answer should clearly explain the concept."
    score, feedback, _ = await grade_quiz(
        state={},  # type: ignore[arg-type]
        redis=redis,
        question=qq.question,
        rubric=rubric,
        user_answer=payload.answer,
        topic=qq.topic,
    )

    qq.user_answer = payload.answer
    qq.score = score
    qq.feedback = feedback
    qq.answered_at = datetime.now(timezone.utc)

    if qq.topic:
        sess.learning_state = update_confidence(
            sess.learning_state or {"topics": []}, qq.topic, score
        )

    await db.flush()

    return QuizQuestionResponse(
        id=qq.id,
        topic=qq.topic,
        question=qq.question,
        grading_rubric=qq.grading_rubric,
        score=qq.score,
        feedback=qq.feedback,
        answered_at=qq.answered_at,
    )


@router.get("/sessions/{session_id}/progress", response_model=ProgressResponse)
async def progress(session_id: int, db: AsyncSession = Depends(get_db)) -> ProgressResponse:
    sess = (await db.execute(select(Session).where(Session.id == session_id))).scalar_one_or_none()
    if sess is None:
        raise HTTPException(404, "Session not found")

    n_msgs = (
        await db.execute(
            select(func.count(Message.id)).where(Message.session_id == session_id)
        )
    ).scalar() or 0
    n_quiz = (
        await db.execute(
            select(func.count(QuizQuestion.id)).where(QuizQuestion.session_id == session_id)
        )
    ).scalar() or 0
    total_cost = (
        await db.execute(
            select(func.coalesce(func.sum(Message.cost), 0.0)).where(
                Message.session_id == session_id
            )
        )
    ).scalar() or 0.0

    return ProgressResponse(
        session_id=session_id,
        learning_state=sess.learning_state or {"topics": []},
        total_messages=int(n_msgs),
        total_quiz_questions=int(n_quiz),
        total_cost=round(float(total_cost), 6),
    )
