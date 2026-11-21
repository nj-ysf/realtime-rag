"""
Structured filter extraction: the extension point between a user question and SQL.

RuleBasedQueryParser is deliberately conservative. It only extracts a constraint when
the wording is explicit (a price bound needs a currency, stock needs stock wording) and
returns nothing otherwise, so it never invents a filter. Replace it with an LLM-based
or schema-driven parser later by implementing the QueryParser protocol; RAGService only
depends on the protocol.
"""

import re
from decimal import Decimal
from typing import Protocol

from rag_api.services.filters import ProductFilters


class QueryParser(Protocol):
    def parse(self, question: str) -> ProductFilters: ...


_CUR = r"(?:mad|dhs?|dirhams?|درهم|دراهم)"
_NUM = r"\d[\d.,  ]*?"
_END = r"(?!\w)"

_MAX_WORDS = (
    r"(?:under|below|less than|cheaper than|up to|at most|no more than|max(?:imum)?(?: price)?(?: of)?|within"
    r"|moins de|moins que|jusqu'?[àa]|jusqu’[àa]|au plus|maximum|inf[ée]rieur(?:e)? [àa]|en dessous de"
    r"|أقل من|اقل من|حتى|<=?)"
)
_MIN_WORDS = (
    r"(?:over|above|more than|at least|starting at|from|minimum"
    r"|plus de|plus que|au moins|[àa] partir de|sup[ée]rieur(?:e)? [àa]|au-dessus de"
    r"|أكثر من|اكثر من|على الأقل|>=?)"
)

_MAX_RE = re.compile(rf"(?<!\w){_MAX_WORDS}\s*(?P<num>{_NUM})\s*{_CUR}{_END}", re.IGNORECASE)
_MIN_RE = re.compile(rf"(?<!\w){_MIN_WORDS}\s*(?P<num>{_NUM})\s*{_CUR}{_END}", re.IGNORECASE)
_RANGE_RE = re.compile(
    rf"(?<!\w)(?:between|entre|بين)\s*(?P<a>{_NUM})\s*(?:{_CUR})?\s*(?:and|et|و|to|-|–)\s*(?P<b>{_NUM})\s*{_CUR}{_END}",
    re.IGNORECASE,
)

_OUT_RE = re.compile(
    r"(?<!\w)(?:out[- ]of[- ]stock|sold out|unavailable|rupture(?: de stock)?|[ée]puis[ée]s?|غير متوفر(?:ة)?|نفد)(?!\w)",
    re.IGNORECASE,
)
_IN_RE = re.compile(
    r"(?<!\w)(?:in[- ]stock|en stock|available|disponibles?|متوفر(?:ة)?|متاح(?:ة)?)(?!\w)",
    re.IGNORECASE,
)


def _parse_amount(raw: str) -> Decimal | None:
    """'10000', '10 000', '10,000', '1.500' -> 1500, '99,90' -> 99.90; ambiguous -> None."""
    text = raw.replace(" ", "").replace(" ", "")
    if re.fullmatch(r"\d+", text):
        return Decimal(text)
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", text):
        return Decimal(re.sub(r"[.,]", "", text))
    if re.fullmatch(r"\d+[.,]\d{1,2}", text):
        return Decimal(text.replace(",", "."))
    return None


class RuleBasedQueryParser:

    def parse(self, question: str) -> ProductFilters:
        min_price: Decimal | None = None
        max_price: Decimal | None = None

        range_match = _RANGE_RE.search(question)
        if range_match:
            low = _parse_amount(range_match.group("a"))
            high = _parse_amount(range_match.group("b"))
            if low is not None and high is not None:
                min_price, max_price = min(low, high), max(low, high)
        else:
            max_match = _MAX_RE.search(question)
            if max_match:
                max_price = _parse_amount(max_match.group("num"))
            min_match = _MIN_RE.search(question)
            if min_match:
                min_price = _parse_amount(min_match.group("num"))

        # "out of stock" is checked first: "unavailable" must not read as "available".
        stock = None
        if _OUT_RE.search(question):
            stock = "out_of_stock"
        elif _IN_RE.search(question):
            stock = "in_stock"

        return ProductFilters(min_price=min_price, max_price=max_price, stock=stock)
