from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

import furrow.llm as llm_module
from furrow.agents._util import parse_json_response
from furrow.config import Provider
from furrow.llm import LLMClient
from tests._fakes import make_settings

# Capture the real class before any test patches ``httpx.AsyncClient``; otherwise
# the fakes would call themselves and recurse infinitely.
RealAsyncClient = httpx.AsyncClient


def _handler(
    expected_model: str,
    messages_expected: list[dict[str, str]],
    reply: str,
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.read())
        assert body["model"] == expected_model
        assert body["stream"] is False
        assert body["messages"] == messages_expected
        return httpx.Response(200, json={"message": {"role": "assistant", "content": reply}})

    return handler


def _make_client(handler: Callable[[httpx.Request], httpx.Response]):
    """Build a fake ``httpx.AsyncClient`` that uses *handler* and forwards kwargs
    (notably ``base_url``) to the real client so URLs resolve correctly."""

    class _Fake:
        def __init__(self, **kwargs) -> None:
            self._client = RealAsyncClient(transport=httpx.MockTransport(handler), **kwargs)

        async def __aenter__(self):
            return self._client

        async def __aexit__(self, *exc):
            await self._client.aclose()

    return _Fake


async def test_ollama_complete(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, provider=Provider.OLLAMA, model="llama3")
    monkeypatch.setattr(
        llm_module.httpx,
        "AsyncClient",
        _make_client(_handler("llama3", [{"role": "user", "content": "hi"}], "hello")),
    )
    client = LLMClient(settings=settings)
    out = await client.complete("hi")
    assert out == "hello"


async def test_ollama_complete_with_system(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, provider=Provider.OLLAMA, model="llama3")
    monkeypatch.setattr(
        llm_module.httpx,
        "AsyncClient",
        _make_client(
            _handler(
                "llama3",
                [
                    {"role": "system", "content": "you are x"},
                    {"role": "user", "content": "hello"},
                ],
                "world",
            )
        ),
    )
    client = LLMClient(settings=settings)
    out = await client.complete("hello", system="you are x")
    assert out == "world"


async def test_llmclient_respects_model_override(tmp_path, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        return httpx.Response(200, json={"message": {"content": body["model"]}})

    settings = make_settings(tmp_path, provider=Provider.OLLAMA, model="default-model")
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", _make_client(handler))
    client = LLMClient(settings=settings)
    out = await client.complete("hi", model="override-model")
    assert out == "override-model"


async def test_unsupported_provider_raises(tmp_path):
    settings = make_settings(tmp_path)
    # Force a provider variant that does not match any known branch.
    settings.provider = "not-a-provider"  # type: ignore[assignment]
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hi")


async def test_llmclient_file_ops_workspace(tmp_path):
    settings = make_settings(tmp_path)
    client = LLMClient(settings=settings)
    await client.write_file("rel/nested.txt", "hello")
    assert (tmp_path / "rel" / "nested.txt").read_text() == "hello"
    assert await client.read_file("rel/nested.txt") == "hello"
    assert "rel/nested.txt" in client.list_files()
    assert client.list_files("missing_dir") == []


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a":1}', {"a": 1}),
        ('```json\n{"a":1}\n```', {"a": 1}),
        ('Here is the json:\n{"a":1}\nok', {"a": 1}),
        ("not json", None),
        ('{"a":1} trailing', {"a": 1}),
    ],
)
def test_parse_json_response(raw, expected):
    assert parse_json_response(raw) == expected
