"""
API settings, read from environment variables (and a project-root .env file).

Embedding model and Qdrant location are imported from the ingestion package so the
API can never query with a different model than the one used to build the index.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from rag_api.core.exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Must run before importing ingestion.config, which reads os.environ at import time.
# Real environment variables always win over the .env file.
load_dotenv(PROJECT_ROOT / ".env")

from ingestion.config import EMBEDDING_MODEL, QDRANT_COLLECTION, QDRANT_URL  # noqa: E402


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from None


def _env_optional_int(name: str, default: int | None) -> int | None:
    """Unset -> default; set but empty -> None (explicitly 'use the provider default')."""
    raw = os.getenv(name)
    if raw is None:
        return default
    if raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        raise ConfigurationError(f"{name} must be an integer or empty, got {raw!r}") from None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}") from None


@dataclass(frozen=True)
class Settings:
    # PostgreSQL (source of truth for price, stock, active status)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ecommerce"
    postgres_user: str = "admin"
    postgres_password: str = ""  # intentionally no default: supply via environment
    postgres_connect_timeout: int = 5

    # Qdrant + embeddings (shared with ingestion)
    qdrant_url: str = QDRANT_URL
    qdrant_collection: str = QDRANT_COLLECTION
    embedding_model: str = EMBEDDING_MODEL

    # Which LLM generates the answer: "ollama" (local) or "gemini" (Google API)
    llm_provider: str = "ollama"

    # Gemini (only used when LLM_PROVIDER=gemini)
    gemini_api_key: str = ""  # intentionally no default: supply via environment
    gemini_model: str = "gemini-3.8-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    gemini_timeout_seconds: float = 60.0
    gemini_thinking_budget: int | None = 0  # 0 = no hidden reasoning tokens (faster); None = model default

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_seconds: float = 120.0

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Retrieval
    default_top_k: int = 5
    max_top_k: int = 20
    candidate_multiplier: int = 4      # Qdrant candidates = top_k * multiplier ...
    max_candidates: int = 50           # ... capped here (also used when filters are active)
    min_score: float = 0.25            # cosine similarity floor; 0 disables (tuned on the seed catalog)
    max_question_length: int = 500

    # Logging
    log_level: str = "INFO"
    log_format: str = "text"           # "text" or "json"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            postgres_host=os.getenv("POSTGRES_HOST", cls.postgres_host),
            postgres_port=_env_int("POSTGRES_PORT", cls.postgres_port),
            postgres_db=os.getenv("POSTGRES_DB", cls.postgres_db),
            postgres_user=os.getenv("POSTGRES_USER", cls.postgres_user),
            postgres_password=os.getenv("POSTGRES_PASSWORD", cls.postgres_password),
            postgres_connect_timeout=_env_int("POSTGRES_CONNECT_TIMEOUT", cls.postgres_connect_timeout),
            qdrant_url=QDRANT_URL,
            qdrant_collection=QDRANT_COLLECTION,
            embedding_model=EMBEDDING_MODEL,
            llm_provider=os.getenv("LLM_PROVIDER", cls.llm_provider).strip().lower(),
            gemini_api_key=os.getenv("GEMINI_API_KEY", cls.gemini_api_key),
            gemini_model=os.getenv("GEMINI_MODEL", cls.gemini_model),
            gemini_base_url=os.getenv("GEMINI_BASE_URL", cls.gemini_base_url).rstrip("/"),
            gemini_timeout_seconds=_env_float("GEMINI_TIMEOUT_SECONDS", cls.gemini_timeout_seconds),
            gemini_thinking_budget=_env_optional_int("GEMINI_THINKING_BUDGET", cls.gemini_thinking_budget),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url).rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
            ollama_timeout_seconds=_env_float("OLLAMA_TIMEOUT_SECONDS", cls.ollama_timeout_seconds),
            api_host=os.getenv("API_HOST", cls.api_host),
            api_port=_env_int("API_PORT", cls.api_port),
            default_top_k=_env_int("RAG_DEFAULT_TOP_K", cls.default_top_k),
            max_top_k=_env_int("RAG_MAX_TOP_K", cls.max_top_k),
            candidate_multiplier=_env_int("RAG_CANDIDATE_MULTIPLIER", cls.candidate_multiplier),
            max_candidates=_env_int("RAG_MAX_CANDIDATES", cls.max_candidates),
            min_score=_env_float("RAG_MIN_SCORE", cls.min_score),
            max_question_length=_env_int("RAG_MAX_QUESTION_LENGTH", cls.max_question_length),
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
            log_format=os.getenv("LOG_FORMAT", cls.log_format),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
