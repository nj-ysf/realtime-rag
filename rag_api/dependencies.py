"""Composition root: builds each service once and shares it across requests."""

import logging
from dataclasses import dataclass
from functools import lru_cache

import httpx
from qdrant_client import QdrantClient

from rag_api.config import Settings, get_settings
from rag_api.core.exceptions import ConfigurationError
from rag_api.database.postgres import PostgresDatabase
from rag_api.services.gemini_service import GeminiService
from rag_api.services.llm_service import LLMService
from rag_api.services.product_lookup import ProductLookup
from rag_api.services.query_parser import RuleBasedQueryParser
from rag_api.services.rag_service import RAGService
from rag_api.services.retriever import ProductRetriever

logger = logging.getLogger("rag_api.dependencies")


def build_llm(settings: Settings) -> LLMService | GeminiService:
    """Both backends expose model / generate(messages) / check(), selected by LLM_PROVIDER."""
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise ConfigurationError("LLM_PROVIDER=gemini requires GEMINI_API_KEY to be set")
        return GeminiService(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            base_url=settings.gemini_base_url,
            timeout_seconds=settings.gemini_timeout_seconds,
        )
    if settings.llm_provider == "ollama":
        return LLMService(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    raise ConfigurationError(f"LLM_PROVIDER must be 'ollama' or 'gemini', got {settings.llm_provider!r}")


@dataclass
class AppServices:
    rag: RAGService
    retriever: ProductRetriever
    database: PostgresDatabase
    llm: LLMService | GeminiService


@lru_cache
def get_services() -> AppServices:
    # Imported here so unit tests that never build real services don't pay for torch.
    from ingestion.embeddings.embedding_service import EmbeddingService

    settings = get_settings()

    logger.info("Loading embedding model", extra={"model": settings.embedding_model})
    embedder = EmbeddingService()  # the ONLY embedding model instance in this process

    # Renamed in sentence-transformers 6; keep working on older versions too.
    get_dimension = getattr(
        embedder.model, "get_embedding_dimension", None
    ) or embedder.model.get_sentence_embedding_dimension

    retriever = ProductRetriever(
        client=QdrantClient(url=settings.qdrant_url),
        embedder=embedder,
        collection=settings.qdrant_collection,
        expected_dimension=get_dimension(),
        min_score=settings.min_score,
    )
    database = PostgresDatabase(settings)
    llm = build_llm(settings)
    rag = RAGService(
        retriever=retriever,
        product_lookup=ProductLookup(database),
        llm=llm,
        query_parser=RuleBasedQueryParser(),
        candidate_multiplier=settings.candidate_multiplier,
        max_candidates=settings.max_candidates,
    )
    return AppServices(rag=rag, retriever=retriever, database=database, llm=llm)
