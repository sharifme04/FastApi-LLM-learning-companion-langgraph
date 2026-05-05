"""Analytics endpoint — sessions, messages, materials, cost."""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.message import Message
from app.models.quiz_question import QuizQuestion
from app.models.session import Session
from app.services.cost import cost_summary

logger = logging.getLogger("learning_companion")
router = APIRouter(prefix="/analytics", tags=["Analytics"])


class CostSummary(BaseModel):
    total_cost: float
    total_messages: int
    total_cache_hits: int
    cache_hit_rate: float
    avg_cost_per_message: float


class AnalyticsSummary(BaseModel):
    total_materials: int
    total_chunks: int
    total_sessions: int
    total_messages: int
    total_quiz_questions: int
    cost_summary: CostSummary


@router.get("/summary", response_model=AnalyticsSummary)
async def analytics_summary(db: AsyncSession = Depends(get_db)) -> AnalyticsSummary:
    total_docs = (await db.execute(select(func.count(Document.id)))).scalar() or 0
    total_chunks = (await db.execute(select(func.count(Chunk.id)))).scalar() or 0
    total_sessions = (await db.execute(select(func.count(Session.id)))).scalar() or 0
    total_messages = (await db.execute(select(func.count(Message.id)))).scalar() or 0
    total_quiz = (await db.execute(select(func.count(QuizQuestion.id)))).scalar() or 0

    cs = await cost_summary(db)
    return AnalyticsSummary(
        total_materials=int(total_docs),
        total_chunks=int(total_chunks),
        total_sessions=int(total_sessions),
        total_messages=int(total_messages),
        total_quiz_questions=int(total_quiz),
        cost_summary=CostSummary(**cs),
    )
