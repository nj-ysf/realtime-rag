from typing import Literal

from pydantic import BaseModel


class DependencyStatus(BaseModel):
    status: Literal["ok", "unavailable"]
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    dependencies: dict[str, DependencyStatus]
