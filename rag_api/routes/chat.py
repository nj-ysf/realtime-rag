from fastapi import APIRouter, Depends

from rag_api.dependencies import AppServices, get_services
from rag_api.schemas.chat import ChatRequest, ChatResponse, ProductReference

router = APIRouter(tags=["chat"])


# Plain `def` on purpose: the pipeline is blocking I/O (Qdrant, PostgreSQL, Ollama,
# the embedding model), so FastAPI runs it in its threadpool instead of the event loop.
@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, services: AppServices = Depends(get_services)) -> ChatResponse:
    result = services.rag.answer(request.question, top_k=request.top_k)

    return ChatResponse(
        question=result.question,
        answer=result.answer,
        products=[
            ProductReference(
                id=item.product.id,
                name=item.product.name,
                brand=item.product.brand,
                category=item.product.category,
                price=float(item.product.price),
                currency=item.product.currency,
                stock_quantity=item.product.stock_quantity,
                score=round(item.score, 4),
            )
            for item in result.products
        ],
        applied_filters=result.applied_filters,
    )
