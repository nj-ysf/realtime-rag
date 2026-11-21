import json

import httpx
import pytest

from rag_api.core.exceptions import LLMError
from rag_api.services.llm_service import LLMService

MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def service(handler, model="llama3.1:8b"):
    client = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return LLMService("http://ollama.test", model, client=client)


def test_generates_and_extracts_clean_answer():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["path"] = request.url.path
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "  Hello  \n"}})

    assert service(handler).generate(MESSAGES) == "Hello"
    assert seen["path"] == "/api/chat"
    assert seen["body"]["model"] == "llama3.1:8b"
    assert seen["body"]["stream"] is False
    assert seen["body"]["messages"] == MESSAGES


def test_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMError, match="timed out"):
        service(handler).generate(MESSAGES)


def test_unreachable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMError, match="unreachable"):
        service(handler).generate(MESSAGES)


def test_model_not_pulled():
    with pytest.raises(LLMError, match="ollama pull"):
        service(lambda r: httpx.Response(404, json={"error": "model not found"})).generate(MESSAGES)


def test_server_error():
    with pytest.raises(LLMError, match="HTTP 500"):
        service(lambda r: httpx.Response(500)).generate(MESSAGES)


@pytest.mark.parametrize("body", [{"unexpected": 1}, {"message": {"content": "   "}}, {"message": {}}])
def test_unusable_responses(body):
    with pytest.raises(LLMError):
        service(lambda r: httpx.Response(200, json=body)).generate(MESSAGES)


def test_invalid_json_body():
    with pytest.raises(LLMError):
        service(lambda r: httpx.Response(200, content=b"not json")).generate(MESSAGES)


def test_health_check_model_present_and_missing():
    tags = {"models": [{"name": "llama3.1:8b"}]}
    service(lambda r: httpx.Response(200, json=tags)).check()
    with pytest.raises(LLMError, match="not pulled"):
        service(lambda r: httpx.Response(200, json=tags), model="other:1b").check()
