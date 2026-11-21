from decimal import Decimal

from rag_api.services.filters import ProductFilters
from rag_api.services.prompt_builder import NO_PRODUCTS_BLOCK, SYSTEM_PROMPT, build_messages
from tests.conftest import LAPTOP_ID, make_product


def user_content(messages):
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    return messages[1]["content"]


def test_contains_question_and_current_postgres_values():
    content = user_content(build_messages("Find a laptop", [make_product()], {LAPTOP_ID: 0.87}))
    assert "User question: Find a laptop" in content
    assert "Price: 7850.00 MAD" in content
    assert "IN STOCK (12 units available)" in content
    assert '"ram": "16GB"' in content
    assert "0.87" in content


def test_out_of_stock_is_explicit():
    content = user_content(build_messages("q", [make_product(stock_quantity=0)]))
    assert "OUT OF STOCK (0 units)" in content
    assert "IN STOCK" not in content


def test_no_products_gives_explicit_empty_context():
    content = user_content(build_messages("q", []))
    assert NO_PRODUCTS_BLOCK in content
    assert "[1]" not in content


def test_applied_filters_are_listed_as_exact():
    filters = ProductFilters(max_price=Decimal("5000"))
    content = user_content(build_messages("q", [make_product()], filters=filters))
    assert "price <= 5000" in content
    assert "enforced exactly by the database" in content


def test_without_filters_states_none_were_applied():
    content = user_content(build_messages("q", [make_product()], filters=ProductFilters()))
    assert "no price or stock constraint was extracted" in content


def test_system_prompt_covers_required_behaviours():
    prompt = SYSTEM_PROMPT.lower()
    for fragment in ["never invent", "same language", "french, english, or arabic", "out of stock",
                     "semantic", "not available", "no relevant products", "ignore any instructions"]:
        assert fragment in prompt, fragment
