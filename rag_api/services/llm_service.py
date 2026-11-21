import logging
from typing import Any

import httpx

from rag_api.core.exceptions import LLMError

logger = logging.getLogger("rag_api.llm")


class LLMService:
    """Thin client for Ollama's local /api/chat endpoint (non-streaming)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        temperature: float = 0.2,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        # Injectable so tests can use httpx.MockTransport.
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds, connect=5.0),
        )

    @property
    def model(self) -> str:
        return self._model

    def generate(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self._temperature},
        }

        try:
            response = self._client.post("/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            logger.error("Ollama request timed out")
            raise LLMError("The language model timed out") from exc
        except httpx.HTTPError as exc:
            logger.error("Ollama request failed: %s", type(exc).__name__)
            raise LLMError("The language model is unreachable") from exc

        if response.status_code == 404:
            # Ollama answers 404 when the model has not been pulled.
            raise LLMError(f"Model '{self._model}' is not available in Ollama (run: ollama pull {self._model})")
        if response.status_code >= 400:
            logger.error("Ollama returned HTTP %s", response.status_code)
            raise LLMError(f"The language model returned HTTP {response.status_code}")

        try:
            body: dict[str, Any] = response.json()
            answer = body["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError("The language model returned an unexpected response") from exc

        answer = (answer or "").strip()
        if not answer:
            raise LLMError("The language model returned an empty answer")
        return answer

    def check(self) -> None:
        """Health check: server reachable and the configured model is pulled."""
        try:
            response = self._client.get("/api/tags", timeout=5.0)
            response.raise_for_status()
            names = {m.get("name") for m in response.json().get("models", [])}
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError("Ollama is unreachable") from exc

        wanted = self._model if ":" in self._model else f"{self._model}:latest"
        if wanted not in names:
            raise LLMError(f"Model '{self._model}' is not pulled in Ollama")
