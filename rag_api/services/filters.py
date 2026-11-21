from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

StockFilter = Literal["in_stock", "out_of_stock"]


@dataclass(frozen=True)
class ProductFilters:
    """
    Exact constraints enforced by PostgreSQL (never by the LLM, never by similarity).
    Inactive products are always excluded and are not represented here.
    """

    min_price: Decimal | None = None
    max_price: Decimal | None = None
    stock: StockFilter | None = None

    @property
    def is_empty(self) -> bool:
        return self.min_price is None and self.max_price is None and self.stock is None

    def describe(self) -> list[str]:
        """Human-readable list, shown to the LLM and returned to API clients."""
        applied: list[str] = []
        if self.min_price is not None:
            applied.append(f"price >= {self.min_price}")
        if self.max_price is not None:
            applied.append(f"price <= {self.max_price}")
        if self.stock == "in_stock":
            applied.append("stock_quantity > 0")
        elif self.stock == "out_of_stock":
            applied.append("stock_quantity = 0")
        return applied
