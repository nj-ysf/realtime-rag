from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from rag_api.config import get_settings

_settings = get_settings()

Question = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=_settings.max_question_length),
]


class ChatRequest(BaseModel):
    question: Question
    top_k: int = Field(default=_settings.default_top_k, ge=1, le=_settings.max_top_k)

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "Find me a laptop suitable for programming", "top_k": 5}]
        }
    }


class ProductReference(BaseModel):
    id: str
    name: str
    brand: str | None
    category: str | None
    price: float
    currency: str | None
    stock_quantity: int
    score: float = Field(description="Semantic similarity (cosine) from the vector search")


class ChatResponse(BaseModel):
    question: str
    answer: str
    products: list[ProductReference]
    applied_filters: list[str] = Field(
        description="Exact filters enforced by PostgreSQL. Anything not listed was matched semantically only."
    )
