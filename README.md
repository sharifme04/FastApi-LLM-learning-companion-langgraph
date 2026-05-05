# Personal AI Learning Companion (LangGraph)

A personalised learning assistant. Upload your study materials, then chat with a multi-agent system that:
- **answers** questions using RAG over your materials,
- **assesses** you with quizzes and updates your confidence per topic,
- **recommends** the next topic based on your learning state.

The three agents run inside a **LangGraph state machine**. A rule-based classifier routes each user message to exactly one agent. Learning state (topics, confidence, recent quiz scores) is persisted to PostgreSQL between turns.

This is **Project 5** — the senior-level capstone of the 5-project portfolio. Production reliability is the differentiator: cost-driven Sonnet → Haiku fallback via a Redis flag, DeepEval-style eval framework with CI gates, structured tracing on every chat turn.

---

## The 3-agent state machine

```
                    User message
                         │
                         ▼
                  ┌─────────────────┐
                  │ classify_intent │   rule-based — quiz / recommend / respond
                  └────────┬────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
    ┌──────────────┐ ┌──────────┐ ┌──────────────┐
    │  Responder   │ │ Assessor │ │ Recommender  │
    │ RAG over     │ │ generate │ │ pick next    │
    │ materials    │ │ quiz +   │ │ topic from   │
    │              │ │ rubric   │ │ confidence   │
    └──────┬───────┘ └────┬─────┘ └──────┬───────┘
           │              │              │
           └──────────────┼──────────────┘
                          ▼
              Persist Message + learning_state update
              Update Redis cost-fallback flag if needed
```

Built on LangGraph's `StateGraph` (with a manual sequential fallback if LangGraph isn't installed, so unit tests don't depend on it).

---

## Project layout

```
project-5-learning-companion-langgraph/
├── app/
│   ├── main.py                    FastAPI entry
│   ├── config.py                  pydantic-settings
│   ├── database.py                async SQLAlchemy + pgvector
│   ├── redis_client.py            Redis pool
│   │
│   ├── agents/                    ★★ the LangGraph state machine
│   │   ├── state.py               LearningState TypedDict (LangGraph state schema)
│   │   ├── responder.py           RAG over materials
│   │   ├── assessor.py            quiz generation + grading + confidence update
│   │   ├── recommender.py         next-topic suggestion
│   │   └── graph.py               StateGraph builder + intent classifier + fallback
│   │
│   ├── models/
│   │   ├── document.py            uploaded learning material
│   │   ├── chunk.py               material chunks + embeddings
│   │   ├── session.py             learning session w/ JSON learning_state
│   │   ├── message.py             every chat turn (with cost, model_used, trace_id)
│   │   ├── quiz_question.py       generated quiz Q + user answer + score
│   │   └── eval_metric.py         DeepEval-style per-metric scores
│   │
│   ├── schemas/
│   │   ├── documents.py
│   │   └── sessions.py            ChatRequest/Response, QuizQuestionResponse, EvalReport
│   │
│   ├── services/
│   │   ├── llm_client.py          ★ Anthropic wrapper — Sonnet → Haiku fallback via Redis flag
│   │   ├── session_manager.py     ★ orchestrates: load state → run graph → persist message
│   │   ├── eval_runner.py         ★ DeepEval-style eval over the learning flow
│   │   ├── pdf_parser.py          (reused from P3)
│   │   ├── chunker.py             (reused)
│   │   ├── embedder.py            (reused)
│   │   ├── retriever.py           (reused)
│   │   ├── ingestion.py           (reused)
│   │   └── cost.py                cost tracking on Message rows
│   │
│   ├── routers/
│   │   ├── health.py
│   │   ├── documents.py           POST /documents/upload, list, get, delete
│   │   ├── sessions.py            ★ /sessions, /messages, /quiz, /progress
│   │   ├── analytics.py
│   │   └── evals.py               GET /evals/report, POST /evals/run
│   │
│   └── utils/
│       ├── logging.py             structured JSON logs w/ request_id
│       └── exceptions.py          AppError + handlers
│
├── eval_dataset/golden.json       seed eval items
├── tests/                         16 passing
├── requirements.txt               (includes langgraph + langchain-core)
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml             api (port 8002) + pgvector pg16 + redis 7
├── .env.example
├── README.md
└── BANGLA_FLOW.md
```

---

## API

| Method | Path                                                   | Purpose                                |
|--------|--------------------------------------------------------|----------------------------------------|
| GET    | `/health`                                              | DB + Redis + pgvector status           |
| POST   | `/documents/upload`                                    | Upload a learning material (PDF)       |
| GET    | `/documents`                                           | List materials                         |
| POST   | `/sessions`                                            | Create a new learning session          |
| GET    | `/sessions/{id}`                                       | Get session + learning_state           |
| **POST** | **`/sessions/{id}/message`**                         | **★ Chat — runs the full graph**       |
| GET    | `/sessions/{id}/quiz`                                  | List quiz questions for the session    |
| POST   | `/sessions/{id}/quiz/{q_id}/answer`                    | Submit quiz answer; updates confidence |
| GET    | `/sessions/{id}/progress`                              | Per-session progress dashboard         |
| GET    | `/analytics/summary`                                   | Global analytics                       |
| GET    | `/evals/report`                                        | Latest DeepEval-style metrics          |
| POST   | `/evals/run`                                           | Trigger eval over the golden dataset   |

---

## Cost control & fallback

The cost guardrail has two layers:

1. **Hard cap** — `DAILY_COST_LIMIT` (default $10). Once today's `messages.cost` sum reaches this, requests get `429 Cost limit exceeded`.
2. **Soft fallback** — `COST_FALLBACK_THRESHOLD` (default $5). When today's total exceeds this, the session manager sets a Redis flag `cost_limit_hit=1`. Every subsequent LLM call routes to **`claude-haiku-4-5-20251001`** instead of Sonnet. The flag has a 24 h TTL so it auto-clears.

This is implemented in [app/services/llm_client.py](app/services/llm_client.py) — `select_model()` consults Redis on every call.

If `ANTHROPIC_API_KEY` is unset or a placeholder, `call_anthropic()` returns deterministic stub text — useful for dev and tests, and the rest of the system still exercises end-to-end.

---

## DeepEval-style framework

[app/services/eval_runner.py](app/services/eval_runner.py) runs the full graph against `eval_dataset/golden.json` and scores 4 metrics per item:

- **correctness** — keyword overlap with `ideal_answer`
- **clarity** — length-vs-level heuristic (long answers penalized for `learning_level: beginner`)
- **citation_quality** — does the response cite sources?
- **personalisation** — does the answer reference the learner's level?

Aggregate scores are persisted to `eval_metrics`. CI gate:

```yaml
- run: pytest
- run: |
    metrics=$(curl -fsS -X POST http://localhost:8002/evals/run)
    echo "$metrics"
    python -c "import json,sys; r=json.loads('''$metrics'''); sys.exit(0 if r['passes_thresholds'] else 1)"
```

Defaults: correctness ≥ 0.7 and clarity ≥ 0.7.

The 4-metric design mirrors DeepEval's structure (per-metric rows, threshold per metric); swap the heuristic scorers for `deepeval`'s actual `LLMTestCase` + `GEval` once you want true LLM judges.

---

## Run it

```bash
cp .env.example .env
# set ANTHROPIC_API_KEY=sk-ant-... (optional — works in stub mode without it)

docker compose up -d
curl http://localhost:8002/health
```

Quick sanity check:
```bash
# Create a session
SID=$(curl -s -X POST http://localhost:8002/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice", "title": "ML basics"}' | python -c "import json,sys;print(json.load(sys.stdin)['id'])")

# Chat (the responder will fire)
curl -X POST http://localhost:8002/sessions/$SID/message \
  -H "Content-Type: application/json" \
  -d '{"content": "How does attention work?"}'

# Trigger a quiz (the assessor will fire)
curl -X POST http://localhost:8002/sessions/$SID/message \
  -H "Content-Type: application/json" \
  -d '{"content": "Quiz me on transformers"}'

# Get a recommendation (the recommender will fire)
curl -X POST http://localhost:8002/sessions/$SID/message \
  -H "Content-Type: application/json" \
  -d '{"content": "What should I study next?"}'
```

---

## Tests

**16 / 16 passing.** Coverage:
- Intent classification (quiz / recommend / respond routing)
- Each agent in isolation (responder, assessor, recommender — uses offline LLM stub)
- The full graph end-to-end with both LangGraph and the fallback dispatcher
- Confidence-update math
- Session-level integration: create → chat → quiz → answer → progress
- Health, analytics, eval report endpoints

Mocking strategy:
- `set_embedder` / `set_reranker` injected with deterministic fakes
- SQLite + aiosqlite in tests (with `VectorCompat` cross-dialect column)
- `ANTHROPIC_API_KEY` not needed — `call_anthropic()` returns canned stubs offline

```bash
pip install -r requirements.txt
pytest
```

---

## Why it's portfolio-worthy

- **Real multi-agent system, not just chained prompts.** Three agents, explicit routing, persistent state. Built on a real production library (LangGraph), with a graceful fallback so tests don't depend on it.
- **Production reliability stack** in one place: cost cap, soft fallback to a cheaper model, DeepEval-style CI gate, structured logs, agent traces by `trace_id`.
- **Learning-state model is non-trivial** — confidence updates use exponential weighting (`α·old + β·new`), the recommender reads it, the responder uses it for level calibration. Connects the agents, not just stitches them.

## Talking points

- "I built a LangGraph state machine with three agent nodes — Responder (RAG), Assessor (quiz + grade), Recommender. A cheap rule-based classifier routes each user turn."
- "Confidence per topic uses an EWMA: `new = 0.7·old + 0.3·quiz_score`. The recommender targets low-confidence topics. The responder uses confidence to calibrate explanation depth."
- "Cost-driven model fallback via a Redis flag — once today's spend crosses the threshold, every call switches to Haiku. The flag has a TTL so it auto-clears."
- "Eval framework persists 4 metric rows per run. CI fails if correctness or clarity drop below 0.7. Swappable to real DeepEval LLM judges later — same data shape."
- "Tests stub the LLM and the embedder, use SQLite + FakeRedis. 16 tests run in 1.4 seconds and cover the agent routing, the math, and the route layer."
# FastApi-LLM-learning-companion-langgraph
