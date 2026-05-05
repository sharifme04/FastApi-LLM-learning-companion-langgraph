# প্রজেক্ট ৫ — Personal AI Learning Companion (বাংলা ফ্লো)

পুরো 5-project portfolio-এর senior-level capstone। **LangGraph state machine** দিয়ে ৩টা agent (Responder, Assessor, Recommender) orchestrate করা; persistent learning state (topic-wise confidence); cost-driven Sonnet→Haiku fallback Redis flag দিয়ে; DeepEval-style eval framework CI gate-এ।

---

## ১. প্রজেক্ট সংক্ষেপে

ব্যবহারকারী learning materials (PDF) আপলোড করে, তারপর chat করে। প্রতিটা message intent অনুযায়ী রুট হয়:
- **Question** → **Responder agent** (RAG over materials)
- **"Quiz me"** → **Assessor agent** (quiz generate + grade + confidence update)
- **"What next"** → **Recommender agent** (next topic suggest based on learning_state)

learning_state JSON-এ session-wise persist:
```json
{"topics": [
  {"name": "Linear Algebra", "confidence": 0.85, "last_quiz_score": 0.9},
  {"name": "SVMs", "confidence": 0.4, "last_quiz_score": 0.3}
]}
```

Confidence update formula: `new = α·old + β·quiz_score` (default α=0.7, β=0.3)।

---

## ২. ডিরেক্টরি স্ট্রাকচার

```
project-5-learning-companion-langgraph/
├── app/
│   ├── main.py                    FastAPI entry
│   ├── config.py                  settings + ANTHROPIC_FALLBACK_MODEL
│   ├── database.py                async engine + pgvector
│   ├── redis_client.py            Redis pool
│   │
│   ├── agents/                    ★★ LangGraph state machine (এই project-এর ✶ part)
│   │   ├── state.py               LearningState TypedDict
│   │   ├── responder.py           RAG agent
│   │   ├── assessor.py            quiz generate + grade
│   │   ├── recommender.py         next-topic agent
│   │   └── graph.py               StateGraph builder + intent classifier + fallback
│   │
│   ├── models/
│   │   ├── document.py            uploaded learning material
│   │   ├── chunk.py               (P3 থেকে inherit)
│   │   ├── session.py             ★ NEW — learning session
│   │   ├── message.py             ★ NEW — chat message (cost, model_used, trace_id সহ)
│   │   ├── quiz_question.py       ★ NEW — quiz Q + answer + score
│   │   └── eval_metric.py         ★ NEW — DeepEval-style score
│   │
│   ├── services/
│   │   ├── llm_client.py          ★★ NEW — Anthropic wrapper, Sonnet→Haiku Redis flag fallback
│   │   ├── session_manager.py     ★★ NEW — load state → run graph → save → flag update
│   │   ├── eval_runner.py         ★ NEW — DeepEval-style 4-metric eval
│   │   └── (P3-এর pdf_parser, chunker, embedder, retriever, ingestion, cost preserved)
│   │
│   ├── routers/
│   │   ├── health.py
│   │   ├── documents.py
│   │   ├── sessions.py            ★ NEW — sessions/messages/quiz/progress
│   │   ├── analytics.py
│   │   └── evals.py
│   │
│   └── utils/                     (logging + exceptions, P3 থেকে)
│
├── eval_dataset/golden.json       eval items
├── tests/                         16 passing
├── docker-compose.yml             port 8002 (P3=8000, P4=8001, P5=8002)
└── README.md / BANGLA_FLOW.md
```

---

## ৩. প্রতিটা নতুন ফাইলের কাজ

### Agents — ★★ এই project-এর core

| ফাইল | কাজ |
|------|-----|
| [app/agents/state.py](app/agents/state.py) | `LearningState` TypedDict — LangGraph-এর state schema। session_id, user_message, learning_state, agent_action, retrieved, response_text, quiz_question, recommendation, tokens, cost, model_used, trace_id সব এতে |
| [app/agents/responder.py](app/agents/responder.py) | retrieve() দিয়ে RAG, তারপর Claude-কে context + question + learner_state দিয়ে answer generate। final_message-এ লেখা |
| [app/agents/assessor.py](app/agents/assessor.py) | দুই function: `generate_quiz()` (topic-এর জন্য question + rubric তৈরি) এবং `grade_quiz()` (user answer-এর 0..1 score)। `update_confidence()` — EWMA দিয়ে confidence update |
| [app/agents/recommender.py](app/agents/recommender.py) | learning_state দেখে Claude থেকে next-topic recommendation নেয় |
| [app/agents/graph.py](app/agents/graph.py) | ★★ LangGraph StateGraph: classify → conditional edge → respond/assess/recommend → END। LangGraph না থাকলে manual sequential fallback |

### Services

| ফাইল | কাজ |
|------|-----|
| [app/services/llm_client.py](app/services/llm_client.py) | ★★ `call_anthropic()` wrapper। `select_model()` Redis flag check করে — `cost_limit_hit=1` থাকলে Haiku, otherwise Sonnet। API key না থাকলে stub response (test/dev-এর জন্য) |
| [app/services/session_manager.py](app/services/session_manager.py) | ★★ `chat_turn()` — user message persist → state বানাও → graph run → assistant message persist → learning_state update → daily cost check → Redis flag toggle |
| [app/services/eval_runner.py](app/services/eval_runner.py) | golden.json-এ প্রতিটা item-এ graph run → 4 metric score (correctness/clarity/citation/personalisation) → eval_metrics-এ save |

### Models — নতুন

| ফাইল | কী রাখে |
|------|---------|
| [app/models/session.py](app/models/session.py) | session — user_id, title, **learning_state (JSON)** — topics + confidence + recent scores |
| [app/models/message.py](app/models/message.py) | প্রতিটা chat turn — role (user/assistant), content, agent_action ("respond"/"assess"/"recommend"), sources, tokens, cost, **model_used, trace_id, cache_hit** |
| [app/models/quiz_question.py](app/models/quiz_question.py) | quiz Q — topic, question, rubric, user_answer, score, feedback, answered_at |
| [app/models/eval_metric.py](app/models/eval_metric.py) | DeepEval-style — run_label, metric_name, score, num_samples, detail (per-item breakdown) |

### Routers — নতুন

| ফাইল | endpoint |
|------|----------|
| [app/routers/sessions.py](app/routers/sessions.py) | `POST /sessions`, `GET /sessions/{id}`, `POST /sessions/{id}/message` (★ মূল chat endpoint), `GET /sessions/{id}/quiz`, `POST /sessions/{id}/quiz/{q_id}/answer`, `GET /sessions/{id}/progress` |
| [app/routers/evals.py](app/routers/evals.py) | `GET /evals/report`, `POST /evals/run` |

---

## ৪. পুরো chat ফ্লো

```
[ব্যবহারকারী] POST /sessions/{id}/message {"content": "Quiz me on transformers"}
         │
         ▼
[routers/sessions.py: post_message()]
         │
         ▼
[services/session_manager.py: chat_turn()]
         │
         ├── 1. user message DB-তে save (Message role="user")
         │
         ├── 2. learning_state load (session.learning_state থেকে)
         │
         ├── 3. LearningState dict তৈরি
         │
         ├── 4. ★★ build_graph(db, redis) → runner
         │       │
         │       └── runner(state) চালাও
         │              │
         │              ▼
         │       ┌──────────────┐
         │       │ classify_    │  rule-based: "quiz" hint match
         │       │ intent       │  → agent_action = "assess"
         │       └──────┬───────┘
         │              │
         │       conditional_edge → "assess" branch
         │              │
         │              ▼
         │       ┌──────────────┐
         │       │ assessor:    │  LLM call (or offline stub)
         │       │ generate_quiz│  → quiz_question, quiz_rubric
         │       │              │  → final_message = "📝 Quiz on..."
         │       └──────┬───────┘
         │              │
         │              ▼
         │            END
         │
         ├── 5. assistant Message DB-তে save
         │       (agent_action="assess", model_used, trace_id সহ)
         │
         ├── 6. assessor হলে — QuizQuestion row create (topic + question parse)
         │
         ├── 7. session.learning_state update (যদি agent change করে)
         │
         ├── 8. cost check:
         │       daily_total = sum(messages.cost where created_at >= today)
         │       if daily_total >= COST_FALLBACK_THRESHOLD:
         │           redis.set("cost_limit_hit", "1", ex=86400)
         │       # পরের call-এ select_model() Haiku return করবে
         │
         └── return ChatResponse
```

---

## ৫. Quiz answer + confidence update flow

```
[ব্যবহারকারী] POST /sessions/{id}/quiz/{q_id}/answer {"answer": "..."}
         │
         ▼
[routers/sessions.py: answer_quiz()]
         │
         ├── QuizQuestion row নাও (question + rubric)
         │
         ├── ★ assessor.grade_quiz(question, rubric, user_answer)
         │       │
         │       └── Claude → JSON {score: 0..1, feedback: "..."}
         │
         ├── QuizQuestion update — user_answer, score, feedback, answered_at
         │
         ├── ★ update_confidence(learning_state, topic, score)
         │       new_confidence = α·old + β·score
         │       (default α=0.7, β=0.3)
         │
         ├── session.learning_state save
         │
         └── return QuizQuestionResponse
```

---

## ৬. Cost fallback Redis flag — কীভাবে কাজ করে?

[app/services/llm_client.py](app/services/llm_client.py):

```python
async def select_model(redis):
    flag = await redis.get("cost_limit_hit")
    if flag and str(flag).lower() in ("1", "true"):
        return settings.anthropic_fallback_model  # claude-haiku-4-5
    return settings.anthropic_model  # claude-sonnet-4
```

[app/services/session_manager.py](app/services/session_manager.py)-এর শেষে:

```python
total_today = await daily_cost_total(db)
await set_fallback_flag(redis, total_today >= settings.cost_fallback_threshold)
```

থ্রেশোল্ড পেরোলে flag set, পরের সব LLM call Haiku-তে যাবে। 24h TTL আছে যাতে auto-clear হয়।

---

## ৭. DeepEval-style framework

[app/services/eval_runner.py](app/services/eval_runner.py):

`eval_dataset/golden.json` থেকে items load → প্রতিটা question-এ graph run → ৪ metric score:

| Metric | কী মাপে |
|--------|---------|
| `correctness` | answer-এ ideal_answer-এর word overlap |
| `clarity` | length-vs-level (beginner-এ long answer penalty) |
| `citation_quality` | sources cite করেছে কিনা ([1] marker বা final_sources) |
| `personalisation` | answer-এ learner level reference আছে কিনা |

প্রতিটা metric one row-এ `eval_metrics` table-এ save। CI-তে check:

```yaml
- run: |
    curl -X POST :8002/evals/run | tee out.json
    python -c "import json,sys;r=json.load(open('out.json'));sys.exit(0 if r['passes_thresholds'] else 1)"
```

`passes_thresholds = correctness >= 0.7 and clarity >= 0.7`।

---

## ৮. Test summary

```bash
pytest
```

**১৬টা test, সবগুলো pass:**

- intent classification (quiz/recommend/respond routing)
- update_confidence math
- প্রতিটা agent isolation-এ (responder needs DB chunks, assessor + recommender stub LLM use করে)
- পুরো graph end-to-end (both LangGraph branch ও fallback branch test হয়)
- session integration: create → chat → quiz → answer → progress
- health, analytics, eval report

কোনো network/torch/Postgres লাগে না — FakeEmbedder, FakeReranker, SQLite, FakeRedis।

---

## ৯. Run

```bash
cp .env.example .env
docker compose up -d
curl http://localhost:8002/health   # P5 → port 8002

# Create session + chat
SID=$(curl -s -X POST :8002/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id":"alice","title":"ML basics"}' | python -c "import json,sys;print(json.load(sys.stdin)['id'])")

curl -X POST :8002/sessions/$SID/message \
  -H "Content-Type: application/json" \
  -d '{"content":"How does attention work?"}'

# Quiz flow
curl -X POST :8002/sessions/$SID/message -H "Content-Type: application/json" \
  -d '{"content":"Quiz me on transformers"}'
QID=$(curl -s :8002/sessions/$SID/quiz | python -c "import json,sys;print(json.load(sys.stdin)[0]['id'])")
curl -X POST :8002/sessions/$SID/quiz/$QID/answer -H "Content-Type: application/json" \
  -d '{"answer":"Transformers use self-attention to relate tokens."}'

# Recommendation
curl -X POST :8002/sessions/$SID/message -H "Content-Type: application/json" \
  -d '{"content":"What should I study next?"}'

# Progress dashboard
curl :8002/sessions/$SID/progress
```

---

## ১০. একনজরে — কোন ফাইল কোন কাজের জন্য

| কাজ | ফাইল |
|-----|------|
| **LangGraph state schema** | [app/agents/state.py](app/agents/state.py) |
| **3-agent state machine** | [app/agents/graph.py](app/agents/graph.py) |
| Responder agent | [app/agents/responder.py](app/agents/responder.py) |
| Assessor agent (quiz + grade) | [app/agents/assessor.py](app/agents/assessor.py) |
| Recommender agent | [app/agents/recommender.py](app/agents/recommender.py) |
| **LLM wrapper + Sonnet/Haiku fallback** | [app/services/llm_client.py](app/services/llm_client.py) |
| **Session orchestrator** | [app/services/session_manager.py](app/services/session_manager.py) |
| **DeepEval-style runner** | [app/services/eval_runner.py](app/services/eval_runner.py) |
| Session model + learning_state JSON | [app/models/session.py](app/models/session.py) |
| Chat message model | [app/models/message.py](app/models/message.py) |
| Quiz question model | [app/models/quiz_question.py](app/models/quiz_question.py) |
| Eval metric model | [app/models/eval_metric.py](app/models/eval_metric.py) |
| Sessions/chat/quiz/progress endpoints | [app/routers/sessions.py](app/routers/sessions.py) |
| Eval endpoints | [app/routers/evals.py](app/routers/evals.py) |

---

**সারসংক্ষেপ:** User message → `session_manager.chat_turn` → LearningState load → `graph.runner()` (classify → branch to one of 3 agents) → final_message + agent_action → DB save → confidence/state update → cost flag toggle → response।
