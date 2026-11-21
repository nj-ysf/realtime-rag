from contextlib import contextmanager
from decimal import Decimal

import psycopg
import pytest

from rag_api.core.exceptions import DatabaseError
from rag_api.services.filters import ProductFilters
from rag_api.services.product_lookup import ProductLookup

ID_A = "11111111-1111-4111-8111-111111111101"
ID_B = "22222222-2222-4222-8222-222222222201"
ID_C = "33333333-3333-4333-8333-333333333301"


def row(pid, **kw):
    base = dict(id=pid, name=f"P-{pid[:2]}", description="d", brand="B", category="Cat",
                price=Decimal("100.00"), currency="MAD", stock_quantity=3,
                attributes={"k": "v"}, is_active=True)
    base.update(kw)
    return base


class FakeDatabase:
    """Returns canned rows and records the SQL + parameters it was given."""

    def __init__(self, rows=None, error=None):
        self.rows = rows or []
        self.error = error
        self.executed = []

    @contextmanager
    def connection(self):
        db = self

        class Cursor:
            def fetchall(self):
                return db.rows

        class Conn:
            def execute(self, query, params):
                db.executed.append((query, params))
                if db.error:
                    raise db.error
                return Cursor()

        yield Conn()


def test_preserves_qdrant_relevance_order_not_database_order():
    db = FakeDatabase([row(ID_A), row(ID_B), row(ID_C)])  # the DB returns A, B, C
    result = ProductLookup(db).get_products([ID_C, ID_A, ID_B])
    assert [p.id for p in result] == [ID_C, ID_A, ID_B]


def test_deleted_or_inactive_products_are_dropped():
    db = FakeDatabase([row(ID_B)])  # A and C are gone from PostgreSQL
    result = ProductLookup(db).get_products([ID_A, ID_B, ID_C])
    assert [p.id for p in result] == [ID_B]


def test_query_is_parameterized_and_excludes_inactive():
    db = FakeDatabase()
    ProductLookup(db).get_products([ID_A], ProductFilters(max_price=Decimal("5000"), stock="in_stock"))
    query, params = db.executed[0]
    assert "p.is_active = TRUE" in query
    assert "%(max_price)s" in query and "5000" not in query
    assert ID_A not in query
    assert params["ids"] == [ID_A] and params["max_price"] == Decimal("5000")
    assert "stock_quantity, 0) > 0" in query


def test_out_of_stock_filter():
    db = FakeDatabase()
    ProductLookup(db).get_products([ID_A], ProductFilters(stock="out_of_stock"))
    assert "<= 0" in db.executed[0][0]


def test_invalid_ids_are_skipped_without_querying():
    db = FakeDatabase()
    assert ProductLookup(db).get_products(["not-a-uuid", "'; DROP TABLE products;--"]) == []
    assert db.executed == []


def test_sql_injection_attempt_in_id_never_reaches_sql():
    db = FakeDatabase()
    ProductLookup(db).get_products([ID_A, "x' OR '1'='1"])
    assert "OR '1'" not in db.executed[0][0]
    assert db.executed[0][1]["ids"] == [ID_A]


def test_database_error_is_wrapped():
    db = FakeDatabase(error=psycopg.OperationalError("down"))
    with pytest.raises(DatabaseError):
        ProductLookup(db).get_products([ID_A])


def test_null_attributes_become_empty_dict():
    db = FakeDatabase([row(ID_A, attributes=None)])
    assert ProductLookup(db).get_products([ID_A])[0].attributes == {}
