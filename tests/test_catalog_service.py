from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import psycopg
import pytest

from rag_api.core.exceptions import DatabaseError
from rag_api.services.catalog_service import CatalogService

ID_SYNCED = "11111111-1111-4111-8111-111111111101"
ID_STALE = "11111111-1111-4111-8111-111111111102"
ID_MISSING = "11111111-1111-4111-8111-111111111103"


def row(pid, **kw):
    base = dict(id=pid, name="Phone", brand="B", category="Electronics", price=Decimal("100.00"),
                currency="MAD", stock_quantity=5, is_active=True, updated_at=datetime.now(timezone.utc))
    base.update(kw)
    return base


class FakeDatabase:
    def __init__(self, rows, error=None):
        self.rows, self.error = rows, error

    @contextmanager
    def connection(self):
        db = self

        class Result:
            def __init__(self, data):
                self.data = data

            def fetchall(self):
                return self.data

            def fetchone(self):
                return self.data

        class Conn:
            def execute(self, query, params=None):
                if db.error:
                    raise db.error
                return Result({"n": len(db.rows)} if "count(*)" in query else db.rows)

        yield Conn()


class FakeQdrant:
    def __init__(self, payloads, error=None):
        self.payloads, self.error = payloads, error

    def retrieve(self, collection_name, ids, with_payload, with_vectors):
        if self.error:
            raise self.error
        return [SimpleNamespace(id=i, payload=self.payloads[i]) for i in ids if i in self.payloads]

    def count(self, collection_name, exact):
        return SimpleNamespace(count=len(self.payloads))


def indexed(**kw):
    payload = {"name": "Phone", "price": "100.00", "stock_quantity": 5, "is_active": True}
    payload.update(kw)
    return payload


def test_sync_statuses():
    rows = [row(ID_SYNCED), row(ID_STALE, price=Decimal("120.00"), stock_quantity=2), row(ID_MISSING)]
    qdrant = FakeQdrant({ID_SYNCED: indexed(), ID_STALE: indexed()})  # stale: index still has old price/stock

    snap = CatalogService(FakeDatabase(rows), qdrant, "products").snapshot()

    by_id = {e.id: e for e in snap.entries}
    assert by_id[ID_SYNCED].sync_status == "synced" and by_id[ID_SYNCED].differences == []
    assert by_id[ID_STALE].sync_status == "stale"
    assert by_id[ID_STALE].differences == ["price", "stock_quantity"]
    assert by_id[ID_MISSING].sync_status == "missing"
    assert snap.postgres_count == 3 and snap.qdrant_count == 2 and snap.qdrant_error is None


def test_price_comparison_is_exact_decimal_not_string():
    # Qdrant stores price as a string; "100.0" and "100.00" are the same price.
    qdrant = FakeQdrant({ID_SYNCED: indexed(price="100.0")})
    snap = CatalogService(FakeDatabase([row(ID_SYNCED)]), qdrant, "products").snapshot()
    assert snap.entries[0].sync_status == "synced"


def test_deactivation_not_yet_indexed_is_stale():
    qdrant = FakeQdrant({ID_SYNCED: indexed(is_active=True)})
    snap = CatalogService(FakeDatabase([row(ID_SYNCED, is_active=False)]), qdrant, "products").snapshot()
    assert snap.entries[0].differences == ["is_active"]


def test_qdrant_down_degrades_to_unknown_instead_of_failing():
    qdrant = FakeQdrant({}, error=ConnectionError("refused"))
    snap = CatalogService(FakeDatabase([row(ID_SYNCED)]), qdrant, "products").snapshot()
    assert snap.entries[0].sync_status == "unknown"
    assert snap.qdrant_count is None and snap.qdrant_error == "Vector store unavailable"


def test_database_down_raises_domain_error():
    service = CatalogService(FakeDatabase([], error=psycopg.OperationalError("down")), FakeQdrant({}), "products")
    with pytest.raises(DatabaseError):
        service.snapshot()


def test_empty_catalog():
    snap = CatalogService(FakeDatabase([]), FakeQdrant({}), "products").snapshot()
    assert snap.entries == [] and snap.postgres_count == 0 and snap.qdrant_count == 0
