from fastapi import APIRouter, Depends, Query

from rag_api.dependencies import AppServices, get_services
from rag_api.schemas.catalog import CatalogItem, CatalogResponse

router = APIRouter(tags=["catalog"])


@router.get("/catalog", response_model=CatalogResponse)
def catalog(
    limit: int = Query(default=100, ge=1, le=500),
    services: AppServices = Depends(get_services),
) -> CatalogResponse:
    """Read-only: PostgreSQL products with their Qdrant index sync status (for the web UI)."""
    snapshot = services.catalog.snapshot(limit=limit)
    return CatalogResponse(
        postgres_count=snapshot.postgres_count,
        qdrant_count=snapshot.qdrant_count,
        qdrant_error=snapshot.qdrant_error,
        items=[
            CatalogItem(
                id=e.id,
                name=e.name,
                brand=e.brand,
                category=e.category,
                price=float(e.price),
                currency=e.currency,
                stock_quantity=e.stock_quantity,
                is_active=e.is_active,
                updated_at=e.updated_at,
                sync_status=e.sync_status,
                differences=e.differences,
            )
            for e in snapshot.entries
        ],
    )
