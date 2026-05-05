"""Anthropic client wrapper with cost-driven Sonnet→Haiku fallback.

The fallback flag lives in Redis: `cost_limit_hit=true`. When set, every
call is routed to the cheaper Haiku model. Set by the cost service when
daily total exceeds settings.cost_fallback_threshold.

For tests/dev without an API key, returns deterministic stubs.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic
import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger("learning_companion")
settings = get_settings()

FALLBACK_FLAG_KEY = "cost_limit_hit"


async def select_model(redis: aioredis.Redis | None) -> str:
    """Pick the model — primary unless the fallback flag is set in Redis."""
    if redis is not None:
        try:
            flag = await redis.get(FALLBACK_FLAG_KEY)
            if flag and str(flag).lower() in ("1", "true"):
                return settings.anthropic_fallback_model
        except Exception:
            pass
    return settings.anthropic_model


async def call_anthropic(
    redis: aioredis.Redis | None,
    system: str,
    user_message: str,
    *,
    max_tokens: int = 600,
    json_mode: bool = False,
) -> dict[str, Any]:
    """Call Claude. Returns dict with text, input_tokens, output_tokens, model.

    If no API key is set, returns a deterministic stub so tests/dev work
    without external dependencies.
    """
    api_key = settings.anthropic_api_key
    model = await select_model(redis)

    if not api_key or api_key.startswith("sk-ant-placeholder"):
        # Offline stub
        if json_mode:
            text = '{"answer": "(offline stub answer)", "topics": [], "score": 0.5, "feedback": "(stub)"}'
        else:
            text = f"(offline stub) Echo: {user_message[:80]}"
        return {
            "text": text,
            "input_tokens": 0,
            "output_tokens": 0,
            "model": model + "+offline",
        }

    client = anthropic.AsyncAnthropic(api_key=api_key)
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIError as e:
        logger.error("Anthropic API error: %s", e)
        raise

    text = "".join(b.text for b in response.content if getattr(b, "type", "text") == "text")
    return {
        "text": text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "model": model,
    }


async def set_fallback_flag(redis: aioredis.Redis, enable: bool, ttl_seconds: int = 86400) -> None:
    """Toggle the cost-fallback flag in Redis."""
    if enable:
        await redis.set(FALLBACK_FLAG_KEY, "1", ex=ttl_seconds)
    else:
        # Set to 0 (don't delete — leaves audit trail)
        await redis.set(FALLBACK_FLAG_KEY, "0", ex=ttl_seconds)
