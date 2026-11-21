from decimal import Decimal

import pytest

from rag_api.services.product_lookup import Product

KETTLE_ID = "11111111-1111-4111-8111-111111111101"
LAPTOP_ID = "11111111-1111-4111-8111-111111111104"


def make_product(product_id: str = LAPTOP_ID, **overrides) -> Product:
    values = dict(
        id=product_id,
        name="ThinkPad E14 Gen 5",
        description="Business laptop built for productivity",
        brand="Lenovo",
        category="Electronics",
        price=Decimal("7850.00"),
        currency="MAD",
        stock_quantity=12,
        attributes={"ram": "16GB", "cpu": "Ryzen 7"},
        is_active=True,
    )
    values.update(overrides)
    return Product(**values)


@pytest.fixture
def product():
    return make_product()
