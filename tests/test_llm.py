from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


async def _fast_sleep(seconds: float) -> None:
    return None


class AsyncMockClient:
    """Fake async callable that records calls and supports injecting failures."""

    def __init__(self, return_value: Any = None, side_effect: Any = None) -> None:
        self.return_value = return_value
        self.side_effect = side_effect
        self.call_count = 0
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._sequence: list[Any] | None = None
        if isinstance(side_effect, list):
            self._sequence = list(side_effect)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        self.calls.append((args, kwargs))

        if self._sequence is not None:
            index = self.call_count - 1
            effect = self._sequence[index] if index < len(self._sequence) else self._sequence[-1]
        else:
            effect = self.side_effect

        if effect is None:
            return self.return_value
        if isinstance(effect, BaseException):
            raise effect
        if callable(effect):
            result = effect(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        if asyncio.iscoroutine(effect):
            return await effect
        return effect


def _make_settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {"provider": Provider.ANTHROPIC, "model": "test-model"}
    defaults.update(overrides)
    return Settings(**defaults)


async def test_provider_dispatch_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LLMClient(settings=_make_settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434"))
    fake = AsyncMockClient(return_value="ollama-result")
    monkeypatch.setattr(client, "_complete_ollama", fake)

    result = await client.complete("hello", "system prompt", "llama3")

    assert result == "ollama-result"
    assert fake.call_count == 1
    args, _ = fake.calls[0]
    assert args == ("hello", "system prompt", "llama3")


async def test_complete_ollama_parses_message() -> None:
    client = LLMClient(settings=_make_settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434"))

    fake_resp = MagicMock()
    fake_resp.raise_for_status.return_value = None
    fake_resp.json.return_value = {"message": {"content": "hi"}}

    client._httpx = AsyncMock()
    client._httpx.post.return_value = fake_resp

    result = await client._complete_ollama("hello", "", "llama3")

    assert result == "hi"
    client._httpx.post.assert_awaited_once()


async def test_complete_ollama_missing_message_returns_empty() -> None:
    client = LLMClient(settings=_make_settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434"))

    fake_resp = MagicMock()
    fake_resp.raise_for_status.return_value = None
    fake_resp.json.return_value = {}

    client._httpx = AsyncMock()
    client._httpx.post.return_value = fake_resp

    result = await client._complete_ollama("hello", "", "llama3")

    assert result == ""


async def test_retry_on_transient_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LLMClient(settings=_make_settings(provider=Provider.ANTHROPIC))
    failing = AsyncMockClient(
        side_effect=[openai.APIError("transient", request=MagicMock(), body=None), "recovered"]
    )
    monkeypatch.setattr(client, "_complete_anthropic", failing)
    monkeypatch.setattr("asyncio.sleep", _fast_sleep)

    result = await client.complete("hi", "", "claude-x")

    assert result == "recovered"
    assert failing.call_count == 2


async def test_retry_exhausted_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LLMClient(settings=_make_settings(provider=Provider.ANTHROPIC))
    constant = AsyncMockClient(
        side_effect=openai.APIError("boom", request=MagicMock(), body=None)
    )
    monkeypatch.setattr(client, "_complete_anthropic", constant)
    monkeypatch.setattr("asyncio.sleep", _fast_sleep)

    with pytest.raises(openai.APIError):
        await client.complete("hi", "", "claude-x")

    assert constant.call_count == 3


async def test_retry_ollama_http_error_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LLMClient(settings=_make_settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434"))
    monkeypatch.setattr("asyncio.sleep", _fast_sleep)

    bad_resp = MagicMock()
    bad_resp.raise_for_status.side_effect = httpx.HTTPError("boom")
    bad_resp.json.return_value = {"message": {"content": "bad"}}

    good_resp = MagicMock()
    good_resp.raise_for_status.return_value = None
    good_resp.json.return_value = {"message": {"content": "recovered"}}

    client._httpx = AsyncMock()
    client._httpx.post.side_effect = [bad_resp, good_resp]

    result = await client.complete("hi", "", "ollama-model")

    assert result == "recovered"
    assert client._httpx.post.await_count == 2
