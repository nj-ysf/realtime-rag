"""
End-to-end tests against the REAL stack (Qdrant, PostgreSQL, embedding model, the configured LLM: Ollama or Gemini).

    pytest -m e2e -v

Reads the same environment/.env as the API (POSTGRES_*, LLM_PROVIDER, OLLAMA_* or GEMINI_*, QDRANT_URL ...).
Each test is skipped, with a reason, when a dependency is unreachable. Read-only:
nothing is written to PostgreSQL or Qdrant.

Assumes the seed catalog from infrastructure/postgres/init.sql has been ingested.
"""

import pytest
from fastapi.testclient import TestClient

from rag_api.dependencies import get_services
from rag_api.main import app
from rag_api.services.filters import ProductFilters

pytestmark = pytest.mark.e2e

THINKPAD_ID = "11111111-1111-4111-8111-111111111104"
INACTIVE_SUNSCREEN_ID = "55555555-5555-4555-8555-555555555502"  # is_active = FALSE in the seed data


@pytest.fixture(scope="module")
def services():
    return get_services()


def require(probe, what):
    try:
        probe()
    except Exception as exc:
        pytest.skip(f"{what} not available: {exc}")


@pytest.fixture(scope="module")
def pg_ready(services):
    require(services.database.ping, "PostgreSQL")


@pytest.fixture(scope="module")
def qdrant_ready(services):
    require(services.retriever.ping, "Qdrant")


@pytest.fixture(scope="module")
def llm_ready(services):
    require(services.llm.check, "LLM (configured provider + model)")


def test_query_embedding_has_expected_dimension(services):
    vector = services.retriever.embed_query("laptop for programming")
    assert len(vector) == 384


def test_semantic_search_finds_relevant_products(services, qdrant_ready):
    results = services.retriever.search("laptop suitable for programming", limit=5)
    assert results, "no results: has the seed catalog been ingested?"
    assert results[0].product_id == THINKPAD_ID
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_inactive_products_are_never_retrieved(services, qdrant_ready):
    results = services.retriever.search("sunscreen fluid SPF50 for the face", limit=20)
    assert INACTIVE_SUNSCREEN_ID not in [r.product_id for r in results]


def test_postgres_lookup_is_authoritative_and_ordered(services, pg_ready):
    from rag_api.services.product_lookup import ProductLookup

    lookup = ProductLookup(services.database)
    ids = [THINKPAD_ID, INACTIVE_SUNSCREEN_ID, "99999999-9999-4999-8999-999999999999"]
    products = lookup.get_products(ids)
    assert [p.id for p in products] == [THINKPAD_ID]  # inactive and unknown ids drop out
    assert products[0].category == "Electronics"
    assert products[0].currency == "MAD"

    assert lookup.get_products([THINKPAD_ID], ProductFilters(max_price=1)) == []


def test_chat_endpoint_with_exact_price_filter(services, pg_ready, qdrant_ready, llm_ready):
    client = TestClient(app)
    response = client.post("/chat", json={"question": "Show me laptops under 5000 MAD", "top_k": 5})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"].strip()
    assert body["applied_filters"] == ["price <= 5000"]
    assert all(p["price"] <= 5000 for p in body["products"])  # enforced by SQL, not by the LLM
    assert THINKPAD_ID not in [p["id"] for p in body["products"]]  # 7850 MAD


def test_chat_endpoint_semantic_question(services, pg_ready, qdrant_ready, llm_ready):
    client = TestClient(app)
    response = client.post("/chat", json={"question": "Find me a laptop suitable for programming", "top_k": 3})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["products"] and body["products"][0]["id"] == THINKPAD_ID
    assert body["applied_filters"] == []
    assert body["answer"].strip()


def test_health_reports_all_dependencies(services, pg_ready, qdrant_ready, llm_ready):
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert {k: v["status"] for k, v in response.json()["dependencies"].items()} == {
        "postgres": "ok", "qdrant": "ok", "llm": "ok",
    }
