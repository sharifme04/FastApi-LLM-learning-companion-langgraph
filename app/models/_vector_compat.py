"""Cross-dialect vector column.

Uses pgvector's Vector type on Postgres and a JSON-encoded fallback on
SQLite (so the same models can be used in tests without a real Postgres).
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import JSON, Dialect, types

try:
    from pgvector.sqlalchemy import Vector as PgVector
except ImportError:  # pgvector not installed
    PgVector = None  # type: ignore[assignment]


class VectorCompat(types.TypeDecorator):
    """Vector column that becomes pgvector.Vector on Postgres, JSON on others."""

    impl = JSON
    cache_ok = True

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql" and PgVector is not None:
            return dialect.type_descriptor(PgVector(self.dim))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value  # pgvector accepts list/np.ndarray directly
        # JSON fallback
        if hasattr(value, "tolist"):
            value = value.tolist()
        return value if isinstance(value, str) else json.dumps(value)

    def process_result_value(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        if isinstance(value, str):
            return json.loads(value)
        return value
