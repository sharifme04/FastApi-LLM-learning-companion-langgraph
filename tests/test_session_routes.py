"""Session-level integration tests."""

import pytest


@pytest.mark.asyncio
async def test_create_session_and_chat(client):
    create = await client.post("/sessions", json={"user_id": "u1", "title": "Algebra study"})
    assert create.status_code == 201, create.text
    sess = create.json()
    session_id = sess["id"]

    msg = await client.post(
        f"/sessions/{session_id}/message",
        json={"content": "How does attention work?"},
    )
    assert msg.status_code == 200, msg.text
    body = msg.json()
    assert body["session_id"] == session_id
    assert body["role"] == "assistant"
    assert body["agent_action"] == "respond"


@pytest.mark.asyncio
async def test_quiz_flow(client):
    create = await client.post("/sessions", json={"user_id": "u1"})
    sid = create.json()["id"]

    quiz_msg = await client.post(
        f"/sessions/{sid}/message", json={"content": "Quiz me on transformers"}
    )
    assert quiz_msg.status_code == 200
    assert quiz_msg.json()["agent_action"] == "assess"

    quiz_list = await client.get(f"/sessions/{sid}/quiz")
    assert quiz_list.status_code == 200
    qs = quiz_list.json()
    assert len(qs) >= 1
    qid = qs[0]["id"]

    ans = await client.post(
        f"/sessions/{sid}/quiz/{qid}/answer",
        json={"answer": "Transformers use self-attention to relate tokens."},
    )
    assert ans.status_code == 200, ans.text
    body = ans.json()
    assert body["score"] is not None
    assert 0.0 <= body["score"] <= 1.0


@pytest.mark.asyncio
async def test_progress_endpoint(client):
    create = await client.post("/sessions", json={"user_id": "u1"})
    sid = create.json()["id"]
    await client.post(f"/sessions/{sid}/message", json={"content": "Hello"})

    p = await client.get(f"/sessions/{sid}/progress")
    assert p.status_code == 200
    body = p.json()
    assert body["session_id"] == sid
    assert body["total_messages"] >= 1


@pytest.mark.asyncio
async def test_analytics_summary(client):
    r = await client.get("/analytics/summary")
    assert r.status_code == 200
    assert "cost_summary" in r.json()


@pytest.mark.asyncio
async def test_eval_report_empty(client):
    r = await client.get("/evals/report")
    assert r.status_code == 200
    body = r.json()
    assert "metrics" in body
    assert body["passes_thresholds"] is False
