import asyncio

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient, _retry


@pytest.mark.asyncio
async def test_ollama_provider_posts_to_openai_compatible_endpoint(monkeypatch):
    settings = Settings(
        provider=Provider.OLLAMA, ollama_base_url="http://ollama.test:11434"
    )
    client = LLMClient(settings=settings)

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "hello"}}]}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            self.url = None
            self.payload = None

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args) -> bool:
            return False

        async def post(self, url: str, json: dict | None = None) -> "FakeResponse":
            self.url = url
            self.payload = json
            return FakeResponse()

    captured: dict[str, FakeAsyncClient] = {}

    def factory(*args, **kwargs) -> FakeAsyncClient:
        instance = FakeAsyncClient(*args, **kwargs)
        captured["client"] = instance
        return instance

    monkeypatch.setattr("furrow.llm.httpx.AsyncClient", factory)

    result = await client.complete("hi", system="sys", model="llama3")

    assert result == "hello"
    assert captured["client"].url == "http://ollama.test:11434/v1/chat/completions"
    assert captured["client"].payload["model"] == "llama3"
    assert captured["client"].payload["messages"][0] == {
        "role": "system",
        "content": "sys",
    }
    assert captured["client"].payload["messages"][1] == {"role": "user", "content": "hi"}


@pytest.mark.asyncio
async def test_retry_decorator_retries_transient_errors():
    state = {"calls": 0}

    @_retry
    async def flaky() -> str:
        state["calls"] += 1
        if state["calls"] < 3:
            raise ConnectionError("transient")
        return "ok"

    assert await flaky() == "ok"
    assert state["calls"] == 3


@pytest.mark.asyncio
async def test_retry_decorator_reraises_after_exhaustion():
    state = {"calls": 0}

    @_retry
    async def always_fails() -> str:
        state["calls"] += 1
        raise ConnectionError("persistent")

    with pytest.raises(ConnectionError):
        await always_fails()
    assert state["calls"] == 3


def test_unknown_provider_raises():
    settings = Settings(provider=Provider.ANTHROPIC)
    settings.provider = "not-a-provider"  # type: ignore[assignment]
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="Unsupported provider"):
        asyncio.run(client.complete("hi"))