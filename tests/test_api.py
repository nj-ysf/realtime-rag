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


class FakeCatalog:
    def __init__(self, snapshot=None, error=None):
        self.snapshot_value, self.error, self.limit = snapshot, error, None

    def snapshot(self, limit):
        self.limit = limit
        if self.error:
            raise self.error
        return self.snapshot_value


def install(rag=None, pg=None, qdrant=None, ollama=None, catalog=None):
    services = SimpleNamespace(
        rag=rag or FakeRAG(ok_result()),
        database=SimpleNamespace(ping=probe(pg)),
        retriever=SimpleNamespace(ping=probe(qdrant)),
        llm=SimpleNamespace(check=probe(ollama)),
        catalog=catalog or FakeCatalog(),
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


# ---------- web UI + catalog ----------

def test_ui_is_served_with_security_headers():
    response = client.get("/ui")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "E-commerce RAG" in response.text
    csp = response.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp


@pytest.mark.parametrize("path,content_type", [
    ("/static/app.js", "javascript"), ("/static/styles.css", "text/css"), ("/static/favicon.svg", "image/svg+xml"),
])
def test_static_assets(path, content_type):
    response = client.get(path)
    assert response.status_code == 200 and content_type in response.headers["content-type"]


def test_ui_has_no_external_dependencies():
    html = client.get("/ui").text
    assert "http://" not in html and "https://" not in html  # no CDN: works offline, leaks nothing


def test_root_links_the_ui():
    assert client.get("/").json()["ui"] == "/ui"


def test_catalog_endpoint_shape():
    from datetime import datetime, timezone
    from decimal import Decimal

    from rag_api.services.catalog_service import CatalogEntry, CatalogSnapshot

    entry = CatalogEntry(
        id=make_product().id, name="ThinkPad", brand="Lenovo", category="Electronics", price=Decimal("7850.00"),
        currency="MAD", stock_quantity=12, is_active=True, updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        sync_status="stale", differences=["price"],
    )
    fake = FakeCatalog(CatalogSnapshot([entry], postgres_count=21, qdrant_count=20))
    install(catalog=fake)

    body = client.get("/catalog?limit=50").json()
    assert fake.limit == 50
    assert body["postgres_count"] == 21 and body["qdrant_count"] == 20 and body["qdrant_error"] is None
    assert body["items"][0]["price"] == 7850.0
    assert body["items"][0]["sync_status"] == "stale" and body["items"][0]["differences"] == ["price"]


@pytest.mark.parametrize("limit", [0, 501, "x"])
def test_catalog_limit_validation(limit):
    install()
    assert client.get(f"/catalog?limit={limit}").status_code == 422


def test_catalog_database_failure_is_503():
    install(catalog=FakeCatalog(error=DatabaseError("Could not read the catalog from PostgreSQL")))
    response = client.get("/catalog")
    assert response.status_code == 503 and "catalog" in response.json()["detail"]
