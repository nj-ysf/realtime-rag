import logging
import math
import uuid
from dataclasses import dataclass
from typing import Protocol, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag_api.core.exceptions import EmbeddingError, RetrievalError

logger = logging.getLogger("rag_api.retriever")

# Inactive products must never be recommended. Postgres re-checks this too.
ACTIVE_ONLY = Filter(must=[FieldCondition(key="is_active", match=MatchValue(value=True))])


class Embedder(Protocol):
    """Satisfied by ingestion.embeddings.embedding_service.EmbeddingService."""

    def generate_embedding(self, text: str) -> Sequence[float]: ...


@dataclass(frozen=True)
class RetrievedCandidate:
    product_id: str
    score: float


class ProductRetriever:
    """Semantic search over the Qdrant `products` collection. Returns IDs and scores only."""

    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        collection: str,
        expected_dimension: int | None = None,
        min_score: float = 0.0,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._collection = collection
        self._expected_dimension = expected_dimension
        self._min_score = min_score

    def embed_query(self, question: str) -> list[float]:
        try:
            vector = [float(x) for x in self._embedder.generate_embedding(question)]
        except Exception as exc:
            logger.error("Query embedding failed: %s", type(exc).__name__)
            raise EmbeddingError("Could not generate an embedding for the question") from exc

        if not vector:
            raise EmbeddingError("The embedding model returned an empty vector")
        if not all(math.isfinite(x) for x in vector):
            raise EmbeddingError("The embedding contains NaN or infinite values")
        if self._expected_dimension is not None and len(vector) != self._expected_dimension:
            raise EmbeddingError(
                f"Embedding has {len(vector)} dimensions, expected {self._expected_dimension}"
            )
        return vector

    def search(
        self,
        question: str,
        limit: int = 5,
        apply_score_floor: bool = True,
    ) -> list[RetrievedCandidate]:
        """
        apply_score_floor=False is used when exact filters (price/stock) are active: the
        question may carry little semantic content ("products under 1000 MAD"), so the
        floor would drop valid candidates before PostgreSQL gets to apply the filter.
        """
        if not question.strip():
            raise ValueError("question must not be empty")
        if limit < 1:
            raise ValueError("limit must be >= 1")

        vector = self.embed_query(question)

        try:
            response = self._client.query_points(
                collection_name=self._collection,
                query=vector,
                query_filter=ACTIVE_ONLY,
                limit=limit,
                with_payload=False,
                score_threshold=(self._min_score or None) if apply_score_floor else None,
            )
        except Exception as exc:
            logger.error("Qdrant query failed: %s", type(exc).__name__)
            raise RetrievalError("Vector search is unavailable") from exc

        candidates: list[RetrievedCandidate] = []
        for point in response.points:
            try:
                product_id = str(uuid.UUID(str(point.id)))
            except ValueError:
                logger.warning("Ignoring Qdrant point with non-UUID id", extra={"point_id": point.id})
                continue
            candidates.append(RetrievedCandidate(product_id=product_id, score=float(point.score)))

        logger.info(
            "Semantic search completed",
            extra={"requested": limit, "returned": len(candidates)},
        )
        return candidates

    def ping(self) -> None:
        try:
            self._client.get_collection(self._collection)
        except Exception as exc:
            raise RetrievalError("Qdrant health check failed") from exc
