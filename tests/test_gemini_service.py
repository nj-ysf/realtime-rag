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
    return GeminiService(SECRET, model, client=client, sleep=lambda _: None)  # no real waits on retries


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
    (400, "API key"), (403, "API key"), (404, "is not available"), (429, "quota"),
    (500, "temporarily unavailable"),
])
def test_http_errors_are_mapped_without_leaking_the_key(status, expected):
    body = {"error": {"status": "INVALID_ARGUMENT", "message": "API key not valid"}}
    with pytest.raises(LLMError, match=expected) as info:
        service(lambda r: httpx.Response(status, json=body)).generate(MESSAGES)
    assert SECRET not in str(info.value)


def test_server_error_without_details():
    with pytest.raises(LLMError, match="HTTP 500"):
        service(lambda r: httpx.Response(500, content=b"")).generate(MESSAGES)


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


# ---------- thinking budget + retries ----------

def retrying_service(handler, **kw):
    client = httpx.Client(base_url="https://gemini.test", transport=httpx.MockTransport(handler))
    sleeps = []
    svc = GeminiService(SECRET, "gemini-x", client=client, sleep=sleeps.append, **kw)
    return svc, sleeps


def test_thinking_budget_is_sent_by_default_as_zero():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=answer("ok"))

    retrying_service(handler)[0].generate(MESSAGES)
    assert seen["body"]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


def test_thinking_budget_none_uses_model_default():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=answer("ok"))

    retrying_service(handler, thinking_budget=None)[0].generate(MESSAGES)
    assert "thinkingConfig" not in seen["body"]["generationConfig"]


def test_overload_is_retried_with_backoff_then_succeeds():
    calls = []
    overloaded = {"error": {"code": 503, "status": "UNAVAILABLE", "message": "This model is currently experiencing high demand."}}

    def handler(request):
        calls.append(1)
        return httpx.Response(503, json=overloaded) if len(calls) < 3 else httpx.Response(200, json=answer("finally"))

    svc, sleeps = retrying_service(handler)
    assert svc.generate(MESSAGES) == "finally"
    assert len(calls) == 3 and sleeps == [1.0, 2.0]


def test_persistent_overload_surfaces_googles_message():
    overloaded = {"error": {"code": 503, "status": "UNAVAILABLE", "message": "This model is currently experiencing high demand."}}
    svc, sleeps = retrying_service(lambda r: httpx.Response(503, json=overloaded))
    with pytest.raises(LLMError, match="temporarily unavailable: This model is currently experiencing high demand"):
        svc.generate(MESSAGES)
    assert len(sleeps) == 2  # max_retries


def test_non_transient_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404, json={"error": {"status": "NOT_FOUND", "message": "gone"}})

    svc, sleeps = retrying_service(handler)
    with pytest.raises(LLMError):
        svc.generate(MESSAGES)
    assert len(calls) == 1 and sleeps == []


def test_unsupported_thinking_setting_points_to_the_env_var():
    body = {"error": {"status": "INVALID_ARGUMENT", "message": "Thinking budget is not supported for this model."}}
    svc, _ = retrying_service(lambda r: httpx.Response(400, json=body))
    with pytest.raises(LLMError, match="GEMINI_THINKING_BUDGET"):
        svc.generate(MESSAGES)


def test_thinking_budget_env_parsing(monkeypatch):
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", "")
    assert Settings.from_env().gemini_thinking_budget is None
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", "512")
    assert Settings.from_env().gemini_thinking_budget == 512
    monkeypatch.delenv("GEMINI_THINKING_BUDGET")
    assert Settings.from_env().gemini_thinking_budget == 0
