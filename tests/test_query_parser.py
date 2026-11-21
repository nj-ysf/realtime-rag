from decimal import Decimal

import pytest

from rag_api.services.query_parser import RuleBasedQueryParser

parser = RuleBasedQueryParser()


@pytest.mark.parametrize("question,expected", [
    ("Show laptops under 10000 MAD", ["price <= 10000"]),
    ("moins de 5 000 dh en stock", ["price <= 5000", "stock_quantity > 0"]),
    ("phones between 3000 and 5000 MAD", ["price >= 3000", "price <= 5000"]),
    ("ordinateur plus de 8.000 dhs", ["price >= 8000"]),
    ("هواتف أقل من 5000 درهم", ["price <= 5000"]),
    ("Which products are currently out of stock?", ["stock_quantity = 0"]),
    ("Which Dell laptops are available?", ["stock_quantity > 0"]),
    ("is this unavailable?", ["stock_quantity = 0"]),
])
def test_extracts_explicit_constraints(question, expected):
    assert parser.parse(question).describe() == expected


@pytest.mark.parametrize("question", [
    "Do you have a gaming laptop with 16GB RAM?",
    "laptop under 16GB RAM",          # a number without a currency is not a price
    "Find a smartphone with a good camera",
    "price 1,5.0 under 1.2.3 mad",    # ambiguous number: never guessed
])
def test_does_not_invent_filters(question):
    assert parser.parse(question).is_empty


def test_decimal_amounts():
    assert parser.parse("under 99,90 MAD").max_price == Decimal("99.90")
