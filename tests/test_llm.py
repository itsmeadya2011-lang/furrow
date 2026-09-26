from __future__ import annotations

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient, LLMError


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    async def flaky(prompt, system, model):
        calls["n"] += 1
        if calls["n"] < 3:
            raise LLMError("transient")
        return "ok"

    client = LLMClient(settings=Settings(retry_max_attempts=3))
    result = await client._with_retry(flaky, "hi", "", "m")
    assert result == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_retry_disabled_when_attempts_one():
    calls = {"n": 0}

    async def fn(prompt, system, model):
        calls["n"] += 1
        return "ok"

    client = LLMClient(settings=Settings(retry_max_attempts=1))
    result = await client._with_retry(fn, "hi", "", "m")
    assert result == "ok"
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_retry_reraises_after_exhaustion():
    calls = {"n": 0}

    async def always_fail(prompt, system, model):
        calls["n"] += 1
        raise LLMError("nope")

    client = LLMClient(settings=Settings(retry_max_attempts=3))
    with pytest.raises(LLMError):
        await client._with_retry(always_fail, "hi", "", "m")
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_complete_ollama_not_supported_without_server(monkeypatch):
    """Ollama path should build a payload and call the httpx client; verify wiring."""
    captured = {}

    class FakeResp:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    class FakeHttp:
        def __init__(self, *a, **k):
            pass

        async def post(self, url, json=None):
            captured["url"] = url
            captured["body"] = json
            return FakeResp({"message": {"content": "hello from ollama"}})

    monkeypatch.setattr(LLMClient, "httpx_client", property(lambda self: FakeHttp()))
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA, model="phi3"))
    text = await client.complete("hi", system="you", model="phi3")
    assert text == "hello from ollama"
    assert captured["url"].endswith("/api/chat")
    assert captured["body"]["model"] == "phi3"
    assert captured["body"]["messages"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_complete_unsupported_provider_raises():
    # Create settings whose provider is forced to an invalid value via object.__setattr__
    # isn't trivial for str-Enum; instead verify the OLLAMA branch wires (covered above)
    # and that an obviously-bad enum value raises ValueError.
    class FakeProvider(str):
        pass

    settings = Settings(provider=Provider.ANTHROPIC)
    # Force provider to an unknown sentinel by bypassing validation.
    object.__setattr__(settings, "provider", "bogus")
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hi")
