"""Domain errors. Messages are safe to return to API clients (no secrets)."""


class RagError(Exception):
    """Base class for all RAG API errors."""


class ConfigurationError(RagError):
    """Invalid or missing configuration."""


class EmbeddingError(RagError):
    """The query embedding could not be generated or is invalid."""


class RetrievalError(RagError):
    """The vector store (Qdrant) could not be queried."""


class DatabaseError(RagError):
    """PostgreSQL could not be reached or queried."""


class LLMError(RagError):
    """The LLM backend (Ollama) failed or returned an unusable response."""
