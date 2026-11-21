import json

import httpx
import pytest

from rag_api.config import Settings
from rag_api.core.exceptions import ConfigurationError, LLMError
from rag_api.dependencies import build_llm
from rag_api.services.gemini_service import GeminiService
from rag_api.services.llm_service import LLMService

SECRET = "AIza-SECRET-KEY-VALUE"
MESSAGES = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}]


def service(handler, model="gemini-2.5-flash"):
    client = httpx.Client(
        base_url="https://gemini.test",
        headers={"x-goog-api-key": SECRET},
        transport=httpx.MockTransport(handler),
    )
    return GeminiService(SECRET, model, client=client)


def answer(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}


def test_request_shape_and_answer_extraction():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["key_header"] = request.headers["x-goog-api-key"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=answer("  Hi there \n"))

    assert service(handler).generate(MESSAGES) == "Hi there"
    assert seen["url"] == "https://gemini.test/v1beta/models/gemini-2.5-flash:generateContent"
    assert SECRET not in seen["url"]              # key travels in a header, not the URL
    assert seen["key_header"] == SECRET
    assert seen["body"]["system_instruction"] == {"parts": [{"text": "be brief"}]}
    assert seen["body"]["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert "temperature" in seen["body"]["generationConfig"]


def test_multiple_parts_are_joined():
    body = {"candidates": [{"content": {"parts": [{"text": "A "}, {"text": "B"}]}}]}
    assert service(lambda r: httpx.Response(200, json=body)).generate(MESSAGES) == "A B"


@pytest.mark.parametrize("status,expected", [
    (400, "API key"), (403, "API key"), (404, "is not available"), (429, "quota"), (500, "HTTP 500"),
])
def test_http_errors_are_mapped_without_leaking_the_key(status, expected):
    body = {"error": {"status": "INVALID_ARGUMENT", "message": "API key not valid"}}
    with pytest.raises(LLMError, match=expected) as info:
        service(lambda r: httpx.Response(status, json=body)).generate(MESSAGES)
    assert SECRET not in str(info.value)


def test_timeout_and_unreachable():
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    def down(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMError, match="timed out"):
        service(slow).generate(MESSAGES)
    with pytest.raises(LLMError, match="unreachable"):
        service(down).generate(MESSAGES)


def test_blocked_prompt():
    body = {"promptFeedback": {"blockReason": "SAFETY"}}
    with pytest.raises(LLMError, match="blocked.*SAFETY"):
        service(lambda r: httpx.Response(200, json=body)).generate(MESSAGES)


@pytest.mark.parametrize("body", [
    {}, {"candidates": []}, {"candidates": [{"content": {"parts": [{"text": "  "}]}, "finishReason": "MAX_TOKENS"}]},
    {"candidates": [{"finishReason": "SAFETY"}]},
])
def test_empty_or_missing_answers(body):
    with pytest.raises(LLMError):
        service(lambda r: httpx.Response(200, json=body)).generate(MESSAGES)


def test_retired_model_surfaces_googles_explanation():
    body = {"error": {"status": "NOT_FOUND", "message": (
        "This model models/gemini-2.5-flash is no longer available to new users. Please update your code "
        "to use models/gemini-3.8-flash for the latest features. We recommend you to use the Interactions API")}}
    with pytest.raises(LLMError) as info:
        service(lambda r: httpx.Response(404, json=body)).generate(MESSAGES)
    message = str(info.value)
    assert "no longer available to new users" in message and "gemini-3.8-flash" in message
    assert "We recommend" not in message  # trimmed marketing tail


def test_health_check_uses_free_count_tokens():
    seen = {}

    def handler(request):
        seen["method"], seen["path"] = request.method, request.url.path
        return httpx.Response(200, json={"totalTokens": 2})

    service(handler).check()
    # countTokens (not GET /models/x): a GET succeeds even for models retired for new projects.
    assert seen == {"method": "POST", "path": "/v1beta/models/gemini-2.5-flash:countTokens"}

    with pytest.raises(LLMError, match="not available"):
        service(lambda r: httpx.Response(404, json={}), model="nope").check()


def test_provider_selection():
    assert isinstance(build_llm(Settings(llm_provider="ollama")), LLMService)
    gemini = build_llm(Settings(llm_provider="gemini", gemini_api_key=SECRET, gemini_model="gemini-x"))
    assert isinstance(gemini, GeminiService) and gemini.model == "gemini-x"


def test_gemini_without_key_fails_fast_with_clear_message():
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        build_llm(Settings(llm_provider="gemini", gemini_api_key=""))


def test_unknown_provider_is_rejected():
    with pytest.raises(ConfigurationError, match="LLM_PROVIDER"):
        build_llm(Settings(llm_provider="openai"))
