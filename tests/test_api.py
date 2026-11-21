from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rag_api.core.exceptions import DatabaseError, LLMError, RetrievalError
from rag_api.dependencies import get_services
from rag_api.main import app
from rag_api.services.rag_service import RAGResult, RankedProduct
from tests.conftest import make_product


class FakeRAG:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def answer(self, question, top_k):
        self.calls.append((question, top_k))
        if self.error:
            raise self.error
        return self.result


def ok_result(question="Find me a laptop"):
    return RAGResult(question, "Here are some laptops", [RankedProduct(make_product(), 0.87)], ["price <= 10000"])


def probe(error):
    def run():
        if error:
            raise error
    return run


def install(rag=None, pg=None, qdrant=None, ollama=None):
    services = SimpleNamespace(
        rag=rag or FakeRAG(ok_result()),
        database=SimpleNamespace(ping=probe(pg)),
        retriever=SimpleNamespace(ping=probe(qdrant)),
        llm=SimpleNamespace(check=probe(ollama)),
    )
    app.dependency_overrides[get_services] = lambda: services
    return services


@pytest.fixture(autouse=True)
def _clean_overrides():
    yield
    app.dependency_overrides.clear()


# Not used as a context manager, so the startup hook (model loading) does not run.
client = TestClient(app, raise_server_exceptions=False)


def test_root():
    body = client.get("/").json()
    assert body["name"] and body["endpoints"]["chat"] == "POST /chat"


def test_chat_success_response_shape():
    rag = FakeRAG(ok_result())
    install(rag=rag)
    response = client.post("/chat", json={"question": "  Find me a laptop  ", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "Find me a laptop"
    assert body["answer"] == "Here are some laptops"
    assert body["applied_filters"] == ["price <= 10000"]
    assert body["products"][0] == {
        "id": make_product().id, "name": "ThinkPad E14 Gen 5", "brand": "Lenovo", "category": "Electronics",
        "price": 7850.0, "currency": "MAD", "stock_quantity": 12, "score": 0.87,
    }
    assert rag.calls == [("Find me a laptop", 3)]  # whitespace stripped before reaching the pipeline


def test_top_k_defaults_to_5():
    rag = FakeRAG(ok_result())
    install(rag=rag)
    client.post("/chat", json={"question": "laptop"})
    assert rag.calls[0][1] == 5


@pytest.mark.parametrize("payload", [
    {"question": ""}, {"question": "   "}, {"question": "x" * 501}, {"question": "ok", "top_k": 0},
    {"question": "ok", "top_k": 21}, {"question": "ok", "top_k": "many"}, {}, {"top_k": 3},
])
def test_request_validation(payload):
    rag = FakeRAG(ok_result())
    install(rag=rag)
    assert client.post("/chat", json=payload).status_code == 422
    assert rag.calls == []  # the pipeline is never reached


@pytest.mark.parametrize("error,status", [
    (RetrievalError("Vector search is unavailable"), 503),
    (DatabaseError("Could not connect to PostgreSQL"), 503),
    (LLMError("The language model timed out"), 502),
])
def test_dependency_failures_map_to_http_errors(error, status):
    install(rag=FakeRAG(error=error))
    response = client.post("/chat", json={"question": "laptop"})
    assert response.status_code == status
    assert response.json() == {"detail": str(error)}


def test_health_all_ok():
    install()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert set(response.json()["dependencies"]) == {"postgres", "qdrant", "llm"}


def test_health_reports_degraded_dependency():
    install(ollama=LLMError("Ollama is unreachable"))
    response = client.get("/health")
    body = response.json()
    assert response.status_code == 503 and body["status"] == "degraded"
    assert body["dependencies"]["llm"] == {"status": "unavailable", "error": "Ollama is unreachable"}
    assert body["dependencies"]["postgres"]["status"] == "ok"


def test_health_never_leaks_unexpected_error_details():
    install(pg=RuntimeError("postgresql://admin:SECRET@host/db"))
    text = client.get("/health").text
    assert "SECRET" not in text and "admin" not in text
    assert "RuntimeError" in text
