"""Pytest fixtures.

- SQLite (aiosqlite) replaces Postgres. The retriever has a SQLite
  fallback that does cosine in Python, so unit tests don't need pgvector.
- A deterministic FakeEmbedder returns hash-bucket vectors.
- A deterministic FakeReranker scores by character overlap.
- Anthropic SDK is patched per test that needs it.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.main import app
from app.redis_client import get_redis
from app.services import embedder as embedder_module


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
test_async_session = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest_asyncio.fixture(autouse=True)
async def _setup_database():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
    async with test_async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


class FakeRedis:
    def __init__(self) -> None:
        self._s: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._s.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._s[key] = value

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


_fake_redis = FakeRedis()


async def _override_get_redis() -> AsyncGenerator[FakeRedis, None]:
    yield _fake_redis


app.dependency_overrides[get_db] = _override_get_db
app.dependency_overrides[get_redis] = _override_get_redis


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    async with test_async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


# ---------- Deterministic fake embedder + reranker ---------- #


class FakeEmbedder:
    """Hash-based deterministic embedder. Same text → same vector."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode()).digest()
            for j in range(self.dim):
                out[i, j] = (h[j % len(h)] / 255.0) - 0.5
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


class FakeReranker:
    """Score by character-level Jaccard overlap of query and candidate."""

    def predict(self, pairs: list[list[str]]) -> np.ndarray:
        out = np.zeros(len(pairs), dtype=np.float32)
        for i, (q, c) in enumerate(pairs):
            qs = set(q.lower().split())
            cs = set(c.lower().split())
            if not qs or not cs:
                out[i] = 0.0
            else:
                out[i] = float(len(qs & cs) / len(qs | cs))
        return out


@pytest.fixture(autouse=True)
def patch_models():
    """Replace the singleton embedder and reranker for every test."""
    embedder_module.set_embedder(FakeEmbedder())
    embedder_module.set_reranker(FakeReranker())
    yield
    embedder_module.set_embedder(None)
    embedder_module.set_reranker(None)


# ---------- Anthropic patcher ---------- #


def make_anthropic_response(text: str, input_tokens: int = 100, output_tokens: int = 50):
    block = MagicMock()
    block.text = text
    block.type = "text"
    resp = MagicMock()
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return resp


@pytest.fixture
def patch_anthropic():
    started: list[Any] = []

    def install(text: str, input_tokens: int = 100, output_tokens: int = 50):
        client = MagicMock()
        client.messages = MagicMock()
        client.messages.create = AsyncMock(
            return_value=make_anthropic_response(text, input_tokens, output_tokens)
        )
        p = patch("anthropic.AsyncAnthropic", return_value=client)
        started.append(p.start())
        return started[-1]

    yield install
    patch.stopall()
