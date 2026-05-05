"""Agent + state-machine tests (uses offline LLM stub via no API key)."""

import pytest

from app.agents.assessor import generate_quiz, update_confidence
from app.agents.graph import classify_intent, build_graph
from app.agents.recommender import recommend
from app.agents.responder import respond
from app.agents.state import LearningState
from app.models.chunk import Chunk
from app.models.document import Document
from app.services.embedder import embed_query


def _empty_state(msg: str) -> LearningState:
    return {
        "session_id": 1,
        "user_message": msg,
        "user_id": "u",
        "learning_state": {"topics": []},
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": 0.0,
        "model_used": "",
        "cache_hit": False,
        "retrieved": [],
        "final_message": "",
        "final_sources": [],
    }


def test_classify_quiz_intent():
    s = _empty_state("Quiz me on attention mechanisms")
    out = classify_intent(s)
    assert out["agent_action"] == "assess"


def test_classify_recommend_intent():
    s = _empty_state("What should I study next?")
    out = classify_intent(s)
    assert out["agent_action"] == "recommend"


def test_classify_default_to_respond():
    s = _empty_state("How does attention work?")
    out = classify_intent(s)
    assert out["agent_action"] == "respond"


def test_update_confidence_new_topic():
    state = update_confidence({"topics": []}, "Algebra", 0.8)
    assert state["topics"][0]["name"] == "Algebra"
    assert state["topics"][0]["last_quiz_score"] == 0.8


def test_update_confidence_existing_topic():
    state = {"topics": [{"name": "Algebra", "confidence": 0.5}]}
    state = update_confidence(state, "Algebra", 1.0, alpha=0.5, beta=0.5)
    assert state["topics"][0]["confidence"] == 0.75


@pytest.mark.asyncio
async def test_responder_returns_message(db_session, fake_redis):
    # Seed one chunk so retrieval has something
    doc = Document(filename="x.pdf", title="X",
                   file_size_bytes=10, total_chunks=1, text_length=100)
    db_session.add(doc)
    await db_session.flush()
    db_session.add(
        Chunk(
            document_id=doc.id, chunk_index=0,
            text="Attention computes weighted sums.",
            embedding=embed_query("Attention computes weighted sums."),
            token_count=5, page_start=1, page_end=1,
        )
    )
    await db_session.flush()

    s = _empty_state("How does attention work?")
    out = await respond(s, db_session, fake_redis)
    assert out["final_message"]
    assert out["model_used"]


@pytest.mark.asyncio
async def test_assessor_generates_quiz(fake_redis):
    s = _empty_state("Quiz me on transformers")
    out = await generate_quiz(s, fake_redis)
    assert "Quiz" in out["final_message"]
    assert out["quiz_question"]


@pytest.mark.asyncio
async def test_recommender_runs(fake_redis):
    s = _empty_state("What should I study next?")
    s["learning_state"] = {
        "topics": [
            {"name": "algebra", "confidence": 0.4},
            {"name": "calculus", "confidence": 0.9},
        ]
    }
    out = await recommend(s, fake_redis)
    assert isinstance(out["final_message"], str)


@pytest.mark.asyncio
async def test_graph_routes_correctly(db_session, fake_redis):
    runner = build_graph(db_session, fake_redis)

    s1 = _empty_state("Quiz me on python")
    out1 = await runner(s1)
    assert out1["agent_action"] == "assess"

    s2 = _empty_state("What should I study next?")
    out2 = await runner(s2)
    assert out2["agent_action"] == "recommend"
