"""Application configuration via pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "learning-companion"
    app_version: str = "1.0.0"
    debug: bool = False

    # PostgreSQL (pgvector + tsvector)
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/learning_companion"
    )

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Anthropic — primary + Haiku fallback (cost-triggered)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_fallback_model: str = "claude-haiku-4-5-20251001"
    anthropic_input_price_per_mtok: float = 3.00
    anthropic_output_price_per_mtok: float = 15.00

    # Embedding + reranker
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_dim: int = 384

    # Retrieval
    top_k_vector: int = 10
    top_k_rerank: int = 5
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 100

    # Cost / rate / limits
    daily_cost_limit: float = 10.00
    cost_fallback_threshold: float = 5.00  # switch to Haiku above this
    rate_limit: str = "30/minute"
    max_pdf_bytes: int = 50 * 1024 * 1024
    max_material_bytes: int = 50 * 1024 * 1024

    # Upload dir
    upload_dir: str = "uploads"

    # Eval (DeepEval-style)
    eval_correctness_threshold: float = 0.7
    eval_clarity_threshold: float = 0.7

    # Quiz / learning state
    quiz_questions_per_topic: int = 3
    confidence_alpha: float = 0.7   # weight on prior confidence
    confidence_beta: float = 0.3    # weight on new score


@lru_cache
def get_settings() -> Settings:
    return Settings()
