from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings
from furrow.llm import LLMClient


class TextBlock:
    def __init__(self, type: str, text: str) -> None:
        self.type = type
        self.text = text


class ThinkingBlock:
    def __init__(self, type: str, thinking: str) -> None:
        self.type = type
        self.thinking = thinking


class FakeMessage:
    def __init__(self, content: object) -> None:
        self.content = content


class FakeMessages:
    def __init__(self, content: object) -> None:
        self.content = content

    async def create(self, **kwargs: object) -> FakeMessage:
        return FakeMessage(self.content)


class FakeAnthropic:
    def __init__(self, content: object) -> None:
        self.messages = FakeMessages(content)


class RateLimitError(Exception):
    """Fake transient error; the retry predicate matches on the class name."""


async def _noop_sleep(*args: object, **kwargs: object) -> None:
    return None


async def test_ollama_provider_posts_to_api(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["url"] = str(request.url)
        sent["method"] = request.method
        sent["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "the answer"}})

    # ``furrow.llm.httpx`` is the ``httpx`` module itself, so capture the real
    # client class before patching ``AsyncClient`` to avoid recursion.
    real_async_client = httpx.AsyncClient

    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr("furrow.llm.httpx.AsyncClient", factory)

    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
    result = await client.complete("write code", system="be helpful")
    assert result == "the answer"
    assert sent["method"] == "POST"
    assert sent["url"] == f"{client.settings.ollama_base_url.rstrip('/')}/api/chat"
    body = sent["body"]
    assert isinstance(body, dict)
    assert body["model"] == client.settings.ollama_model
    assert body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "write code"},
    ]


async def test_ollama_explicit_model_is_sent_in_body(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "the answer"}})

    real_async_client = httpx.AsyncClient

    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr("furrow.llm.httpx.AsyncClient", factory)

    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
    result = await client.complete("write code", system="be helpful", model="custom-model")
    assert result == "the answer"
    body = sent["body"]
    assert isinstance(body, dict)
    assert body["model"] == "custom-model"


async def test_anthropic_joins_only_text_blocks() -> None:
    client = LLMClient(
        settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    )
    client._anthropic = FakeAnthropic(
        [
            TextBlock(type="text", text="a"),
            ThinkingBlock(type="thinking", thinking="..."),
            TextBlock(type="text", text="b"),
        ]
    )
    assert await client.complete("prompt") == "a\nb"


async def test_anthropic_returns_empty_when_there_are_no_text_blocks() -> None:
    client = LLMClient(
        settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    )
    client._anthropic = FakeAnthropic([ThinkingBlock(type="thinking", thinking="...")])
    assert await client.complete("prompt") == ""


async def test_retry_on_transient_error_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    attempts = 0

    async def fake(prompt: str, system: str, model: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RateLimitError("slow down")
        return "done"

    client = LLMClient(
        settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    )
    client._complete_anthropic = fake
    assert await client.complete("prompt") == "done"
    assert attempts == 3


async def test_non_retryable_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    attempts = 0

    async def fake(prompt: str, system: str, model: str) -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("missing API key")

    client = LLMClient(
        settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    )
    client._complete_anthropic = fake
    with pytest.raises(ValueError, match="missing API key"):
        await client.complete("prompt")
    assert attempts == 1


async def test_settings_none_falls_back_to_the_default_settings() -> None:
    assert LLMClient().settings is default_settings


async def test_openai_path_returns_message_content() -> None:
    client = LLMClient(settings=Settings(provider=Provider.OPENAI, openai_api_key="test-key"))
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))])

    async def create(**kwargs: object) -> SimpleNamespace:
        return response

    client._openai = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    assert await client.complete("prompt") == "hello"


async def test_list_files_and_read_write_file_helpers(tmp_path: Path) -> None:
    client = LLMClient()
    await client.write_file(tmp_path / "a.txt", "hello")
    assert await client.read_file(tmp_path / "a.txt") == "hello"
    assert sorted(client.list_files(tmp_path)) == ["a.txt"]
    assert client.list_files(tmp_path / "does-not-exist") == []
