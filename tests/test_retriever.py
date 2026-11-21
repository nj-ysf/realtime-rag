from types import SimpleNamespace

import pytest

from rag_api.core.exceptions import EmbeddingError, RetrievalError
from rag_api.services.retriever import ACTIVE_ONLY, ProductRetriever

ID_A = "11111111-1111-4111-8111-111111111101"
ID_B = "22222222-2222-4222-8222-222222222201"


class FakeEmbedder:
    def __init__(self, vector=None, error=None):
        self.vector = vector if vector is not None else [0.1, 0.2, 0.3]
        self.error = error
        self.calls = []

    def generate_embedding(self, text):
        self.calls.append(text)
        if self.error:
            raise self.error
        return self.vector


class FakeQdrant:
    def __init__(self, points=None, error=None):
        self.points = points or []
        self.error = error
        self.kwargs = None

    def query_points(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(points=self.points)

    def get_collection(self, name):
        if self.error:
            raise self.error


def retriever(client, embedder=None, **kw):
    return ProductRetriever(client, embedder or FakeEmbedder(), "products", expected_dimension=3, **kw)


def point(pid, score):
    return SimpleNamespace(id=pid, score=score)


def test_query_embedding_is_generated_from_the_question():
    embedder = FakeEmbedder()
    retriever(FakeQdrant(), embedder).search("find a laptop")
    assert embedder.calls == ["find a laptop"]


def test_search_returns_ids_and_scores_in_qdrant_order():
    client = FakeQdrant([point(ID_A, 0.9), point(ID_B, 0.5)])
    result = retriever(client).search("laptop", limit=2)
    assert [(c.product_id, c.score) for c in result] == [(ID_A, 0.9), (ID_B, 0.5)]


def test_inactive_products_are_filtered_in_the_query():
    client = FakeQdrant()
    retriever(client, min_score=0.3).search("laptop", limit=7)
    assert client.kwargs["query_filter"] is ACTIVE_ONLY
    assert client.kwargs["limit"] == 7
    assert client.kwargs["score_threshold"] == 0.3
    assert client.kwargs["with_payload"] is False  # never trust Qdrant payload for price/stock


def test_score_floor_can_be_disabled_for_filtered_queries():
    client = FakeQdrant()
    retriever(client, min_score=0.3).search("products under 1000 MAD", limit=50, apply_score_floor=False)
    assert client.kwargs["score_threshold"] is None


def test_empty_results():
    assert retriever(FakeQdrant([])).search("anything") == []


def test_points_with_non_uuid_ids_are_ignored():
    client = FakeQdrant([point(42, 0.9), point("not-a-uuid", 0.8), point(ID_A, 0.7)])
    assert [c.product_id for c in retriever(client).search("x")] == [ID_A]


def test_qdrant_connection_error_becomes_retrieval_error():
    with pytest.raises(RetrievalError):
        retriever(FakeQdrant(error=ConnectionError("refused"))).search("x")


@pytest.mark.parametrize("vector", [[], [0.1, float("nan"), 0.3], [0.1, float("inf"), 0.3], [0.1, 0.2]])
def test_invalid_embeddings_are_rejected(vector):
    client = FakeQdrant()
    with pytest.raises(EmbeddingError):
        retriever(client, FakeEmbedder(vector)).search("x")
    assert client.kwargs is None  # Qdrant is never queried with a bad vector


def test_embedding_model_failure_becomes_embedding_error():
    with pytest.raises(EmbeddingError):
        retriever(FakeQdrant(), FakeEmbedder(error=RuntimeError("boom"))).search("x")


def test_blank_question_is_rejected():
    with pytest.raises(ValueError):
        retriever(FakeQdrant()).search("   ")


def test_ping_failure():
    with pytest.raises(RetrievalError):
        retriever(FakeQdrant(error=ConnectionError())).ping()
