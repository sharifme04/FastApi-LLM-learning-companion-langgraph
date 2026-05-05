"""Eval endpoints — list latest metrics, trigger a manual run."""

import logging

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.eval_metric import EvalMetric
from app.redis_client import get_redis
from app.schemas.sessions import EvalReport
from app.services.eval_runner import run_eval

logger = logging.getLogger("learning_companion")
router = APIRouter(prefix="/evals", tags=["Evals"])
settings = get_settings()


def _passes(metrics: dict[str, float]) -> bool:
    return (
        metrics.get("correctness", 0.0) >= settings.eval_correctness_threshold
        and metrics.get("clarity", 0.0) >= settings.eval_clarity_threshold
    )


@router.get("/report", response_model=EvalReport)
async def latest(db: AsyncSession = Depends(get_db)) -> EvalReport:
    rows = (
        await db.execute(
            select(EvalMetric).order_by(EvalMetric.created_at.desc()).limit(8)
        )
    ).scalars().all()
    if not rows:
        return EvalReport(
            metrics={},
            threshold_correctness=settings.eval_correctness_threshold,
            threshold_clarity=settings.eval_clarity_threshold,
            passes_thresholds=False,
            num_samples=0,
            notes="No eval runs yet.",
        )
    latest_run = rows[0].created_at
    metrics_by_name: dict[str, float] = {}
    for r in rows:
        if r.created_at == latest_run and r.metric_name not in metrics_by_name:
            metrics_by_name[r.metric_name] = r.score
    return EvalReport(
        last_run_at=latest_run,
        metrics=metrics_by_name,
        threshold_correctness=settings.eval_correctness_threshold,
        threshold_clarity=settings.eval_clarity_threshold,
        passes_thresholds=_passes(metrics_by_name),
        num_samples=rows[0].num_samples,
    )


@router.post("/run", response_model=EvalReport)
async def run(
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
) -> EvalReport:
    metrics = await run_eval(db, redis)
    return EvalReport(
        metrics=metrics,
        threshold_correctness=settings.eval_correctness_threshold,
        threshold_clarity=settings.eval_clarity_threshold,
        passes_thresholds=_passes(metrics),
        num_samples=0,
    )
