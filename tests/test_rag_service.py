from decimal import Decimal

import pytest

from rag_api.core.exceptions import DatabaseError, LLMError, RetrievalError
from rag_api.services.filters import ProductFilters
from rag_api.services.query_parser import RuleBasedQueryParser
from rag_api.services.rag_service import RAGService
from rag_api.services.retriever import RetrievedCandidate
from tests.conftest import KETTLE_ID, LAPTOP_ID, make_product


class FakeRetriever:
    def __init__(self, candidates=(), error=None):
        self.candidates = list(candidates)
        self.error = error
        self.limit = None
        self.apply_score_floor = None

    def search(self, question, limit, apply_score_floor=True):
        self.limit = limit
        self.apply_score_floor = apply_score_floor
        if self.error:
            raise self.error
        return self.candidates


class FakeLookup:
    def __init__(self, products=(), error=None):
        self.products = list(products)
        self.error = error
        self.args = None

    def get_products(self, ids, filters=None):
        self.args = (list(ids), filters)
        if self.error:
            raise self.error
        # Like PostgreSQL: only existing products come back, in the requested order.
        return [p for i in ids for p in self.products if p.id == i]


class FakeLLM:
    def __init__(self, answer="the answer", error=None):
        self.answer = answer
        self.error = error
        self.messages = None

    def generate(self, messages):
        self.messages = messages
        if self.error:
            raise self.error
        return self.answer


def build(candidates=(), products=(), retriever=None, lookup=None, llm=None):
    retriever = retriever or FakeRetriever(candidates)
    lookup = lookup or FakeLookup(products)
    llm = llm or FakeLLM()
    return RAGService(retriever, lookup, llm, RuleBasedQueryParser()), retriever, lookup, llm


def test_full_workflow():
    cands = [RetrievedCandidate(LAPTOP_ID, 0.87), RetrievedCandidate(KETTLE_ID, 0.40)]
    products = [make_product(LAPTOP_ID), make_product(KETTLE_ID, name="Kettle")]
    service, _, lookup, llm = build(cands, products)

    result = service.answer("Find me a laptop", top_k=2)

    assert result.answer == "the answer"
    assert [(r.product.id, r.score) for r in result.products] == [(LAPTOP_ID, 0.87), (KETTLE_ID, 0.40)]
    assert lookup.args[0] == [LAPTOP_ID, KETTLE_ID]
    assert "ThinkPad E14 Gen 5" in llm.messages[1]["content"]
    assert "Find me a laptop" in llm.messages[1]["content"]


def test_prices_come_from_postgres_not_qdrant():
    service, *_ = build([RetrievedCandidate(LAPTOP_ID, 0.9)], [make_product(price=Decimal("1234.00"))])
    assert service.answer("laptop").products[0].product.price == Decimal("1234.00")


def test_deleted_product_still_in_qdrant_is_ignored():
    cands = [RetrievedCandidate(LAPTOP_ID, 0.9), RetrievedCandidate(KETTLE_ID, 0.8)]
    service, *_ = build(cands, [make_product(KETTLE_ID)])  # the laptop was deleted in PostgreSQL
    assert [r.product.id for r in service.answer("x").products] == [KETTLE_ID]


def test_result_is_trimmed_to_top_k_after_overfetch():
    cands = [RetrievedCandidate(LAPTOP_ID, 0.9), RetrievedCandidate(KETTLE_ID, 0.8)]
    products = [make_product(LAPTOP_ID), make_product(KETTLE_ID)]
    service, retriever, *_ = build(cands, products)
    assert len(service.answer("x", top_k=1).products) == 1
    assert retriever.limit == 4  # top_k * candidate_multiplier


def test_exact_filters_are_enforced_by_postgres_and_reported():
    service, retriever, lookup, llm = build([RetrievedCandidate(LAPTOP_ID, 0.9)], [make_product()])
    result = service.answer("laptops under 10000 MAD in stock", top_k=3)

    assert result.applied_filters == ["price <= 10000", "stock_quantity > 0"]
    assert lookup.args[1] == ProductFilters(max_price=Decimal("10000"), stock="in_stock")
    assert retriever.limit == 50  # widest candidate pool when filters are active
    assert retriever.apply_score_floor is False  # PostgreSQL decides, not the similarity floor
    assert "price <= 10000" in llm.messages[1]["content"]


def test_score_floor_applies_to_purely_semantic_questions():
    service, retriever, *_ = build([RetrievedCandidate(LAPTOP_ID, 0.9)], [make_product()])
    service.answer("a laptop for programming")
    assert retriever.apply_score_floor is True


def test_no_results_still_asks_llm_with_explicit_empty_context():
    service, _, _, llm = build([], [])
    result = service.answer("weather tomorrow")
    assert result.products == []
    assert "No relevant products were found" in llm.messages[1]["content"]


def test_retrieval_failure_propagates_and_stops_the_pipeline():
    lookup, llm = FakeLookup(), FakeLLM()
    service, *_ = build(retriever=FakeRetriever(error=RetrievalError("qdrant down")), lookup=lookup, llm=llm)
    with pytest.raises(RetrievalError):
        service.answer("laptop")
    assert lookup.args is None and llm.messages is None


def test_database_failure_propagates_and_llm_is_not_called():
    llm = FakeLLM()
    service, *_ = build([RetrievedCandidate(LAPTOP_ID, 0.9)], lookup=FakeLookup(error=DatabaseError("pg down")), llm=llm)
    with pytest.raises(DatabaseError):
        service.answer("laptop")
    assert llm.messages is None


def test_llm_failure_propagates():
    service, *_ = build([RetrievedCandidate(LAPTOP_ID, 0.9)], [make_product()], llm=FakeLLM(error=LLMError("down")))
    with pytest.raises(LLMError):
        service.answer("laptop")
