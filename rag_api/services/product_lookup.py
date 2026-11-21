import logging
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Sequence

import psycopg

from rag_api.core.exceptions import DatabaseError
from rag_api.database.postgres import PostgresDatabase
from rag_api.services.filters import ProductFilters

logger = logging.getLogger("rag_api.product_lookup")


@dataclass(frozen=True)
class Product:
    """Authoritative product state as currently stored in PostgreSQL."""

    id: str
    name: str
    description: str | None
    brand: str | None
    category: str | None
    price: Decimal
    currency: str | None
    stock_quantity: int
    attributes: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


_BASE_QUERY = """
    SELECT p.id::text AS id, p.name, p.description, p.brand, c.name AS category,
           p.price, p.currency, COALESCE(p.stock_quantity, 0) AS stock_quantity,
           p.attributes, p.is_active
    FROM products p
    LEFT JOIN categories c ON c.id = p.category_id
    WHERE p.id = ANY(%(ids)s::uuid[])
      AND p.is_active = TRUE
"""


class ProductLookup:
    """Fetches current product data from PostgreSQL, preserving the caller's ID order."""

    def __init__(self, database: PostgresDatabase) -> None:
        self._database = database

    def get_products(
        self,
        product_ids: Sequence[str],
        filters: ProductFilters | None = None,
    ) -> list[Product]:
        valid_ids: list[str] = []
        for raw in product_ids:
            try:
                valid_ids.append(str(uuid.UUID(str(raw))))
            except ValueError:
                logger.warning("Skipping invalid product id", extra={"product_id": raw})

        if not valid_ids:
            return []

        # Only fixed SQL fragments are appended; every value goes through parameters.
        query = _BASE_QUERY
        params: dict[str, Any] = {"ids": valid_ids}

        if filters is not None:
            if filters.min_price is not None:
                query += " AND p.price >= %(min_price)s"
                params["min_price"] = filters.min_price
            if filters.max_price is not None:
                query += " AND p.price <= %(max_price)s"
                params["max_price"] = filters.max_price
            if filters.stock == "in_stock":
                query += " AND COALESCE(p.stock_quantity, 0) > 0"
            elif filters.stock == "out_of_stock":
                query += " AND COALESCE(p.stock_quantity, 0) <= 0"

        try:
            with self._database.connection() as conn:
                rows = conn.execute(query, params).fetchall()
        except psycopg.Error as exc:
            logger.error("Product lookup failed: %s", type(exc).__name__)
            raise DatabaseError("Could not read products from PostgreSQL") from exc

        by_id = {row["id"]: row for row in rows}

        # Qdrant relevance order wins. IDs missing from PostgreSQL (deleted, inactive,
        # or excluded by filters) simply drop out; Qdrant may lag behind the CDC pipeline.
        products: list[Product] = []
        for product_id in valid_ids:
            row = by_id.get(product_id)
            if row is None:
                continue
            products.append(
                Product(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    brand=row["brand"],
                    category=row["category"],
                    price=row["price"],
                    currency=row["currency"],
                    stock_quantity=row["stock_quantity"],
                    attributes=row["attributes"] or {},
                    is_active=row["is_active"],
                )
            )

        dropped = len(valid_ids) - len(products)
        logger.info(
            "Product lookup completed",
            extra={"requested": len(valid_ids), "found": len(products), "dropped": dropped},
        )
        return products
