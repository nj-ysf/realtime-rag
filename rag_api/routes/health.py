from typing import Callable

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from rag_api.core.exceptions import RagError
from rag_api.dependencies import AppServices, get_services
from rag_api.schemas.health import DependencyStatus, HealthResponse

router = APIRouter(tags=["health"])


def _check(probe: Callable[[], None]) -> DependencyStatus:
    # Our own exceptions carry client-safe messages; anything else is reduced to its type name
    # so connection strings or credentials can never leak through this endpoint.
    try:
        probe()
    except RagError as exc:
        return DependencyStatus(status="unavailable", error=str(exc))
    except Exception as exc:
        return DependencyStatus(status="unavailable", error=type(exc).__name__)
    return DependencyStatus(status="ok")


@router.get("/health", response_model=HealthResponse)
def health(services: AppServices = Depends(get_services)) -> JSONResponse:
    dependencies = {
        "postgres": _check(services.database.ping),
        "qdrant": _check(services.retriever.ping),
        "llm": _check(services.llm.check),  # Ollama or Gemini, per LLM_PROVIDER
    }
    healthy = all(d.status == "ok" for d in dependencies.values())
    body = HealthResponse(status="ok" if healthy else "degraded", dependencies=dependencies)
    return JSONResponse(status_code=200 if healthy else 503, content=body.model_dump())
