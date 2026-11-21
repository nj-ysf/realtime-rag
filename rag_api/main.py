import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag_api.config import get_settings
from rag_api.core.exceptions import (
    ConfigurationError,
    DatabaseError,
    EmbeddingError,
    LLMError,
    RagError,
    RetrievalError,
)
from rag_api.core.logging import configure_logging
from rag_api.routes import chat, health

logger = logging.getLogger("rag_api")

# Domain error -> HTTP status. Messages are client-safe by construction.
_ERROR_STATUS: dict[type[RagError], int] = {
    RetrievalError: 503,
    DatabaseError: 503,
    LLMError: 502,
    EmbeddingError: 500,
    ConfigurationError: 500,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    # Load the embedding model at startup rather than on the first user request.
    from rag_api.dependencies import get_services

    services = get_services()
    logger.info("RAG API ready", extra={"llm_provider": settings.llm_provider, "llm_model": services.llm.model})
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Real-Time E-commerce RAG API",
        description="Natural-language product Q&A: Qdrant semantic search + PostgreSQL truth + Ollama.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        logger.info(
            "HTTP request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
        )
        return response

    def _register(error_type: type[RagError], status_code: int) -> None:
        @app.exception_handler(error_type)
        async def handler(request: Request, exc: RagError) -> JSONResponse:
            logger.error("Request failed", extra={"error_type": type(exc).__name__, "path": request.url.path})
            return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    for error_type, status_code in _ERROR_STATUS.items():
        _register(error_type, status_code)

    app.include_router(chat.router)
    app.include_router(health.router)

    @app.get("/", tags=["info"])
    def root() -> dict:
        return {
            "name": "Real-Time E-commerce RAG API",
            "version": app.version,
            "docs": "/docs",
            "endpoints": {"chat": "POST /chat", "health": "GET /health"},
        }

    return app


app = create_app()


if __name__ == "__main__":
    _settings = get_settings()
    uvicorn.run("rag_api.main:app", host=_settings.api_host, port=_settings.api_port)
