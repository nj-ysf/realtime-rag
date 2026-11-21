import logging
import time
from dataclasses import dataclass

from rag_api.services.llm_service import LLMService
from rag_api.services.product_lookup import Product, ProductLookup
from rag_api.services.prompt_builder import build_messages
from rag_api.services.query_parser import QueryParser
from rag_api.services.retriever import ProductRetriever

logger = logging.getLogger("rag_api.rag")


@dataclass(frozen=True)
class RankedProduct:
    product: Product
    score: float


@dataclass(frozen=True)
class RAGResult:
    question: str
    answer: str
    products: list[RankedProduct]
    applied_filters: list[str]


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


class RAGService:
    """
    question -> filters -> Qdrant candidates -> PostgreSQL (truth + exact filters) -> prompt -> LLM.

    Each stage is a separate collaborator; this class only orchestrates them.
    """

    def __init__(
        self,
        retriever: ProductRetriever,
        product_lookup: ProductLookup,
        llm: LLMService,
        query_parser: QueryParser,
        candidate_multiplier: int = 4,
        max_candidates: int = 50,
    ) -> None:
        self._retriever = retriever
        self._lookup = product_lookup
        self._llm = llm
        self._parser = query_parser
        self._candidate_multiplier = candidate_multiplier
        self._max_candidates = max_candidates

    def answer(self, question: str, top_k: int = 5) -> RAGResult:
        started = time.perf_counter()

        # 1. Exact constraints (price / stock). Empty when nothing explicit was found.
        filters = self._parser.parse(question)

        # 2. Semantic candidates. Over-fetch so exact filters applied afterwards in
        #    PostgreSQL can't empty the result just because the top hits were filtered out.
        if filters.is_empty:
            candidate_limit = top_k * self._candidate_multiplier
        else:
            candidate_limit = self._max_candidates
        candidate_limit = min(max(candidate_limit, top_k), self._max_candidates)

        t = time.perf_counter()
        candidates = self._retriever.search(
            question,
            limit=candidate_limit,
            apply_score_floor=filters.is_empty,
        )
        retrieval_ms = _ms(t)

        # 3. Authoritative data + exact filters from PostgreSQL, in Qdrant relevance order.
        t = time.perf_counter()
        products = self._lookup.get_products([c.product_id for c in candidates], filters)[:top_k]
        lookup_ms = _ms(t)

        scores = {c.product_id: c.score for c in candidates}
        ranked = [RankedProduct(product=p, score=scores[p.id]) for p in products]

        # 4. Prompt + generation.
        messages = build_messages(
            question,
            products,
            scores={p.id: scores[p.id] for p in products},
            filters=filters,
        )
        t = time.perf_counter()
        answer = self._llm.generate(messages)
        llm_ms = _ms(t)

        logger.info(
            "RAG request completed",
            extra={
                "question_chars": len(question),
                "top_k": top_k,
                "filters": filters.describe(),
                "candidates": len(candidates),
                "products_used": len(ranked),
                "retrieval_ms": retrieval_ms,
                "lookup_ms": lookup_ms,
                "llm_ms": llm_ms,
                "total_ms": _ms(started),
            },
        )

        return RAGResult(
            question=question,
            answer=answer,
            products=ranked,
            applied_filters=filters.describe(),
        )
