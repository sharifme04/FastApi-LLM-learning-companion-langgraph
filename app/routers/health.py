"""Health check."""

import logging

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.redis_client import get_redis

logger = logging.getLogger("learning_companion")
router = APIRouter(tags=["Health"])
settings = get_settings()


@router.get("/health")
async def health_check(
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
) -> dict:
    health = {
        "status": "ok",
        "version": settings.app_version,
        "service": settings.app_name,
        "db": "disconnected",
        "redis": "disconnected",
        "pgvector": "unknown",
    }
    try:
        await db.execute(text("SELECT 1"))
        health["db"] = "connected"
        if db.bind and db.bind.dialect.name == "postgresql":
            res = await db.execute(text("SELECT extname FROM pg_extension WHERE extname='vector'"))
            health["pgvector"] = "installed" if res.scalar() else "missing"
        else:
            health["pgvector"] = "n/a (sqlite)"
    except Exception as e:
        logger.error("DB health check failed: %s", e)
        health["status"] = "degraded"
        health["db"] = f"error: {str(e)[:100]}"

    try:
        await redis.ping()
        health["redis"] = "connected"
    except Exception as e:
        logger.error("Redis health check failed: %s", e)
        health["status"] = "degraded"
        health["redis"] = f"error: {str(e)[:100]}"

    return health
