import logging
import time
from typing import Any, Callable

import httpx

from rag_api.core.exceptions import LLMError

logger = logging.getLogger("rag_api.llm")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"

# Transient: rate limiting and server-side overload ("model is experiencing high demand").
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GeminiService:
    """
    Google Gemini via the REST API (generateContent). Same interface as LLMService
    (model, generate, check) so RAGService and the health route don't care which is used.

    The API key is sent in the x-goog-api-key header, never in the URL, so it cannot end
    up in request logs; it is also never included in error messages.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = GEMINI_BASE_URL,
        timeout_seconds: float = 60.0,
        temperature: float = 0.2,
        thinking_budget: int | None = 0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._model = model
        self._temperature = temperature
        # 0 = no hidden "thinking" tokens: the prompt already holds every fact, and thinking
        # multiplied latency several times in measurements. None = model default.
        self._thinking_budget = thinking_budget
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff_seconds
        self._sleep = sleep
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"x-goog-api-key": api_key},
            timeout=httpx.Timeout(timeout_seconds, connect=5.0),
        )

    @property
    def model(self) -> str:
        return self._model

    @staticmethod
    def _to_request(
        messages: list[dict[str, str]],
        temperature: float,
        thinking_budget: int | None = None,
    ) -> dict[str, Any]:
        system_parts = [{"text": m["content"]} for m in messages if m["role"] == "system"]
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
            if m["role"] != "system"
        ]
        generation_config: dict[str, Any] = {"temperature": temperature}
        if thinking_budget is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_parts:
            body["system_instruction"] = {"parts": system_parts}
        return body

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return

        try:
            error = response.json().get("error", {})
            reason = error.get("status", "")
            google_message = str(error.get("message", ""))
        except ValueError:
            reason, google_message = "", ""
        logger.error("Gemini returned HTTP %s %s", status, reason)

        if status in (401, 403) or (status == 400 and reason == "INVALID_ARGUMENT" and "API key" in response.text):
            raise LLMError("Gemini rejected the API key (check GEMINI_API_KEY)")
        if status == 404:
            # Google explains *why* (e.g. "no longer available to new users, use models/..."),
            # which is far more useful than a generic message. It never contains the key.
            detail = google_message.split(". We recommend")[0][:300] if google_message else "model not found"
            raise LLMError(f"Gemini model '{self._model}' is not available: {detail} (check GEMINI_MODEL)")
        if status == 400 and "thinking" in google_message.lower():
            raise LLMError(
                f"Gemini model '{self._model}' rejected the thinking setting "
                "(set GEMINI_THINKING_BUDGET to empty to use the model default)"
            )
        if status == 429:
            # Google says which limit was hit (per-minute vs daily, free tier...): keep the first sentence.
            detail = google_message.split("\n")[0].split(". ")[0][:200] if google_message else ""
            raise LLMError(
                "Gemini quota or rate limit exceeded, try again later" + (f" ({detail})" if detail else "")
            )
        if status >= 500 and google_message:
            # e.g. 503 "This model is currently experiencing high demand. ..."
            raise LLMError(f"Gemini is temporarily unavailable: {google_message[:200]}")
        raise LLMError(f"Gemini returned HTTP {status}")

    def _post_with_retry(self, path: str, body: dict[str, Any]) -> httpx.Response:
        """Retries transient failures (429 / 5xx) with exponential backoff. Timeouts are not retried."""
        attempt = 0
        while True:
            try:
                response = self._client.post(path, json=body)
            except httpx.TimeoutException as exc:
                logger.error("Gemini request timed out")
                raise LLMError("Gemini timed out") from exc
            except httpx.HTTPError as exc:
                logger.error("Gemini request failed: %s", type(exc).__name__)
                raise LLMError("Gemini is unreachable") from exc

            if response.status_code not in RETRYABLE_STATUS or attempt >= self._max_retries:
                return response

            delay = self._retry_backoff * (2 ** attempt)
            attempt += 1
            logger.warning(
                "Gemini transient error, retrying",
                extra={"status": response.status_code, "attempt": attempt, "delay_s": delay},
            )
            self._sleep(delay)

    def generate(self, messages: list[dict[str, str]]) -> str:
        response = self._post_with_retry(
            f"/v1beta/models/{self._model}:generateContent",
            self._to_request(messages, self._temperature, self._thinking_budget),
        )
        self._raise_for_status(response)

        try:
            body: dict[str, Any] = response.json()
        except ValueError as exc:
            raise LLMError("Gemini returned an unexpected response") from exc

        candidates = body.get("candidates") or []
        if not candidates:
            block = (body.get("promptFeedback") or {}).get("blockReason")
            raise LLMError(f"Gemini blocked the request ({block})" if block else "Gemini returned no answer")

        parts = (candidates[0].get("content") or {}).get("parts") or []
        answer = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
        if not answer:
            finish = candidates[0].get("finishReason", "unknown")
            raise LLMError(f"Gemini returned an empty answer (finishReason={finish})")
        return answer

    def check(self) -> None:
        """
        Health check: the key is accepted AND this key may use the model for generation.

        Uses countTokens (free, nothing is generated). A plain GET on the model is not
        enough: it succeeds for models that are listed but retired for new projects.
        """
        try:
            response = self._client.post(
                f"/v1beta/models/{self._model}:countTokens",
                json={"contents": [{"role": "user", "parts": [{"text": "health check"}]}]},
                timeout=5.0,
            )
        except httpx.HTTPError as exc:
            raise LLMError("Gemini is unreachable") from exc
        self._raise_for_status(response)
