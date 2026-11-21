from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class CatalogItem(BaseModel):
    id: str
    name: str
    brand: str | None
    category: str | None
    price: float
    currency: str | None
    stock_quantity: int
    is_active: bool
    updated_at: datetime | None
    sync_status: Literal["synced", "stale", "missing", "unknown"] = Field(
        description="synced: Qdrant matches PostgreSQL; stale: indexed values differ; "
                    "missing: not indexed yet; unknown: Qdrant unreachable"
    )
    differences: list[str] = Field(description="Fields whose indexed value differs from PostgreSQL")


class CatalogResponse(BaseModel):
    postgres_count: int
    qdrant_count: int | None
    qdrant_error: str | None
    items: list[CatalogItem]
