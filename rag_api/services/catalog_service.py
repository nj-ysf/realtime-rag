"""
Read-only catalog view: PostgreSQL (source of truth) side by side with the Qdrant index.

Used by the web UI to make the CDC pipeline visible: change a product in PostgreSQL and
watch its status go from "stale" back to "synced" once the ingestion consumer catches up.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

import psycopg
from qdrant_client import QdrantClient

from rag_api.core.exceptions import DatabaseError
from rag_api.database.postgres import PostgresDatabase

logger = logging.getLogger("rag_api.catalog")

SyncStatus = Literal["synced", "stale", "missing", "unknown"]

_LIST_QUERY = """
    SELECT p.id::text AS id, p.name, p.brand, c.name AS category, p.price, p.currency,
           COALESCE(p.stock_quantity, 0) AS stock_quantity, p.is_active, p.updated_at
    FROM products p
    LEFT JOIN categories c ON c.id = p.category_id
    ORDER BY p.updated_at DESC NULLS LAST, p.name
    LIMIT %(limit)s
"""
_COUNT_QUERY = "SELECT count(*) AS n FROM products"


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    name: str
    brand: str | None
    category: str | None
    price: Decimal
    currency: str | None
    stock_quantity: int
    is_active: bool
    updated_at: datetime | None
    sync_status: SyncStatus
    differences: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CatalogSnapshot:
    entries: list[CatalogEntry]
    postgres_count: int
    qdrant_count: int | None
    qdrant_error: str | None = None


def _differences(row: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    """Fields whose indexed (Qdrant) value no longer matches PostgreSQL."""
    diffs: list[str] = []
    if payload.get("name") != row["name"]:
        diffs.append("name")
    try:
        indexed_price = Decimal(str(payload.get("price")))
    except (InvalidOperation, ValueError):
        indexed_price = None
    if indexed_price != row["price"]:
        diffs.append("price")
    if payload.get("stock_quantity") != row["stock_quantity"]:
        diffs.append("stock_quantity")
    if payload.get("is_active") != row["is_active"]:
        diffs.append("is_active")
    return diffs


class CatalogService:

    def __init__(self, database: PostgresDatabase, qdrant: QdrantClient, collection: str) -> None:
        self._database = database
        self._qdrant = qdrant
        self._collection = collection

    def snapshot(self, limit: int = 100) -> CatalogSnapshot:
        try:
            with self._database.connection() as conn:
                rows = conn.execute(_LIST_QUERY, {"limit": limit}).fetchall()
                postgres_count = conn.execute(_COUNT_QUERY).fetchone()["n"]
        except psycopg.Error as exc:
            logger.error("Catalog query failed: %s", type(exc).__name__)
            raise DatabaseError("Could not read the catalog from PostgreSQL") from exc

        # Qdrant problems degrade the view (status "unknown") instead of failing it:
        # PostgreSQL is the source of truth and is still worth showing.
        payloads: dict[str, dict[str, Any]] | None = None
        qdrant_count: int | None = None
        qdrant_error: str | None = None
        try:
            if rows:
                points = self._qdrant.retrieve(
                    collection_name=self._collection,
                    ids=[row["id"] for row in rows],
                    with_payload=["name", "price", "stock_quantity", "is_active"],
                    with_vectors=False,
                )
                payloads = {str(p.id): (p.payload or {}) for p in points}
            else:
                payloads = {}
            qdrant_count = self._qdrant.count(collection_name=self._collection, exact=True).count
        except Exception as exc:
            logger.warning("Qdrant unavailable for catalog view: %s", type(exc).__name__)
            payloads, qdrant_error = None, "Vector store unavailable"

        entries: list[CatalogEntry] = []
        for row in rows:
            if payloads is None:
                status, diffs = "unknown", []
            elif row["id"] not in payloads:
                status, diffs = "missing", []
            else:
                diffs = _differences(row, payloads[row["id"]])
                status = "stale" if diffs else "synced"

            entries.append(CatalogEntry(
                id=row["id"],
                name=row["name"],
                brand=row["brand"],
                category=row["category"],
                price=row["price"],
                currency=row["currency"],
                stock_quantity=row["stock_quantity"],
                is_active=row["is_active"],
                updated_at=row["updated_at"],
                sync_status=status,
                differences=diffs,
            ))

        return CatalogSnapshot(
            entries=entries,
            postgres_count=postgres_count,
            qdrant_count=qdrant_count,
            qdrant_error=qdrant_error,
        )
