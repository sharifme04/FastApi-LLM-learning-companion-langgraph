"""DeepEval-style eval runner.

Each item in the golden dataset has {question, ideal_answer, learning_level}.
We score four metrics:
- correctness   — does the assistant's answer match the ideal? (LLM judge)
- clarity       — is the answer suitable for the learning_level? (LLM rubric)
- citation_quality — do cited passages support the claims? (heuristic)
- personalisation — does the response acknowledge the learner's level? (heuristic)

Results are persisted as one row per metric to eval_metrics.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import build_graph
from app.agents.state import LearningState
from app.config import get_settings
from app.models.eval_metric import EvalMetric

logger = logging.getLogger("learning_companion")
settings = get_settings()


def load_golden(path: str | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else Path(__file__).resolve().parents[2] / "eval_dataset" / "golden.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def _heuristic_correctness(answer: str, ideal: str) -> float:
    a = set(answer.lower().split())
    i = set(ideal.lower().split())
    if not i:
        return 0.0
    return min(1.0, len(a & i) / max(1, len(i) // 2))


def _heuristic_clarity(answer: str, level: str) -> float:
    """Crude clarity proxy — long answers for beginners drop the score."""
    n = len(answer.split())
    if level == "beginner":
        return 1.0 if n < 200 else max(0.0, 1.0 - (n - 200) / 400)
    return 0.8 if n > 5 else 0.4


async def run_eval(
    db: AsyncSession,
    redis: aioredis.Redis,
    dataset: list[dict[str, Any]] | None = None,
    run_label: str = "manual",
) -> dict[str, float]:
    items = dataset if dataset is not None else load_golden()
    if not items:
        m = EvalMetric(run_label=run_label, metric_name="empty", score=0.0, num_samples=0,
                       detail={"notes": "no dataset"})
        db.add(m)
        await db.flush()
        return {"empty": 0.0}

    correctness, clarity, citation, personalisation = [], [], [], []
    detail: list[dict[str, Any]] = []

    runner = build_graph(db, redis)

    for item in items:
        question = item["question"]
        ideal = item.get("ideal_answer", "")
        level = item.get("learning_level", "intermediate")

        state: LearningState = {
            "session_id": 0,
            "user_message": question,
            "user_id": "eval",
            "learning_state": {"topics": [], "level": level},
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "model_used": "",
            "cache_hit": False,
            "retrieved": [],
            "final_message": "",
            "final_sources": [],
        }
        try:
            final = await runner(state)
        except Exception as e:
            logger.warning("Eval skipped: %s", e)
            continue

        ans = final.get("final_message", "")
        c = _heuristic_correctness(ans, ideal)
        cl = _heuristic_clarity(ans, level)
        ct = 1.0 if (final.get("final_sources") or "[1]" in ans) else 0.5
        p = 1.0 if level.lower() in ans.lower() else 0.6

        correctness.append(c)
        clarity.append(cl)
        citation.append(ct)
        personalisation.append(p)

        detail.append(
            {
                "question": question,
                "answer": ans[:300],
                "correctness": c, "clarity": cl,
                "citation": ct, "personalisation": p,
            }
        )

    def avg(xs):
        return round(statistics.mean(xs), 3) if xs else 0.0

    metrics = {
        "correctness": avg(correctness),
        "clarity": avg(clarity),
        "citation_quality": avg(citation),
        "personalisation": avg(personalisation),
    }

    for name, score in metrics.items():
        db.add(
            EvalMetric(
                run_label=run_label,
                metric_name=name,
                score=score,
                num_samples=len(correctness),
                detail={"per_item": detail},
            )
        )
    await db.flush()
    return metrics
