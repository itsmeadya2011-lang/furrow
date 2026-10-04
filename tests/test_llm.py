from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import anthropic
import httpx
import pytest

from furrow import llm as llm_module
from furrow.config import Provider, Settings
from furrow.llm import MAX_TOKENS, LLMClient


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "provider": Provider.ANTHROPIC,
        "model": "test-model",
        "anthropic_api_key": "test-key",
        "openai_api_key": "test-key",
    }
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------------------------- #
# Fakes mimicking the provider SDK shapes without any network access
# --------------------------------------------------------------------------- #


class FakeBlock:
    """Mimics an Anthropic content block."""

    def __init__(self, block_type: str, text: str | None = None) -> None:
        self.type = block_type
        if text is not None:
            self.text = text


class FakeAnthropicResponse:
    def __init__(self, content: list[FakeBlock]) -> None:
        self.content = content


class FakeMessages:
    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self._script) - 1)
        item = self._script[index]
        if isinstance(item, BaseException):
            raise item
        return item


class FakeAnthropic:
    def __init__(self, script: list[Any]) -> None:
        self.messages = FakeMessages(script)


class FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = FakeMessage(content)


class FakeOpenAIResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self._script) - 1)
        item = self._script[index]
        if isinstance(item, BaseException):
            raise item
        return item


class FakeChat:
    def __init__(self, script: list[Any]) -> None:
        self.completions = FakeCompletions(script)


class FakeOpenAI:
    def __init__(self, script: list[Any]) -> None:
        self.chat = FakeChat(script)


class FakeHTTPStatus:
    """Duck-typed stand-in for an httpx response on an SDK status error."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.request = None
        self.headers: dict[str, str] = {}


def status_error(
    error_type: type[Exception], status_code: int, message: str = "boom"
) -> Exception:
    return error_type(message, response=FakeHTTPStatus(status_code), body=None)


class FakeHTTPResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status {self.status_code}", request=None, response=None  # type: ignore[arg-type]
            )

    def json(self) -> Any:
        return self._payload


class FakeAsyncClientFactory:
    """Records requests made through the fake ``httpx.AsyncClient``."""

    requests: ClassVar[list[tuple[str, Any]]] = []
    payload: ClassVar[Any] = {}
    status_code: int = 200

    def __call__(self, *args: Any, **kwargs: Any) -> FakeAsyncClient:
        return FakeAsyncClient(*args, **kwargs)


class FakeAsyncClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_args = args
        self.init_kwargs = kwargs

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False

    async def post(self, url: str, json: Any = None, **kwargs: Any) -> FakeHTTPResponse:
        FakeAsyncClientFactory.requests.append((url, json))
        return FakeHTTPResponse(
            FakeAsyncClientFactory.payload, FakeAsyncClientFactory.status_code
        )


@pytest.fixture
def fast_retries(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make the tenacity backoff effectively instantaneous."""
    monkeypatch.setattr(llm_module, "RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(llm_module, "RETRY_INITIAL_WAIT", 0.0)
    monkeypatch.setattr(llm_module, "RETRY_MAX_WAIT", 0.0)
    monkeypatch.setattr(llm_module, "RETRY_JITTER", 0.0)


@pytest.fixture
def fake_ollama(monkeypatch: pytest.MonkeyPatch) -> type[FakeAsyncClientFactory]:
    FakeAsyncClientFactory.requests = []
    FakeAsyncClientFactory.payload = {"message": {"content": "ollama says hi"}}
    FakeAsyncClientFactory.status_code = 200
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", FakeAsyncClientFactory())
    return FakeAsyncClientFactory


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


async def test_anthropic_joins_multiple_text_blocks() -> None:
    client = LLMClient(settings=make_settings())
    fake = FakeAnthropic(
        [FakeAnthropicResponse([FakeBlock("text", "first"), FakeBlock("text", "second")])]
    )
    client._anthropic = fake  # type: ignore[assignment]

    result = await client.complete("hello")

    assert result == "first\n\nsecond"
    assert fake.messages.calls[0]["model"] == "test-model"
    assert fake.messages.calls[0]["max_tokens"] == MAX_TOKENS
    assert fake.messages.calls[0]["messages"] == [{"role": "user", "content": "hello"}]


async def test_anthropic_skips_non_text_leading_block() -> None:
    client = LLMClient(settings=make_settings())
    fake = FakeAnthropic(
        [
            FakeAnthropicResponse(
                [
                    FakeBlock("thinking"),
                    FakeBlock("tool_use"),
                    FakeBlock("text", "the real answer"),
                ]
            )
        ]
    )
    client._anthropic = fake  # type: ignore[assignment]

    assert await client.complete("hello") == "the real answer"


async def test_anthropic_without_text_block_raises() -> None:
    client = LLMClient(settings=make_settings())
    fake = FakeAnthropic([FakeAnthropicResponse([FakeBlock("thinking")])])
    client._anthropic = fake  # type: ignore[assignment]

    with pytest.raises(ValueError, match="no text content block"):
        await client.complete("hello")
    assert len(fake.messages.calls) == 1


async def test_anthropic_uses_system_prompt() -> None:
    client = LLMClient(settings=make_settings())
    fake = FakeAnthropic([FakeAnthropicResponse([FakeBlock("text", "ok")])])
    client._anthropic = fake  # type: ignore[assignment]

    await client.complete("prompt", system="be terse")

    assert fake.messages.calls[0]["system"] == "be terse"


async def test_anthropic_uses_default_system_prompt() -> None:
    client = LLMClient(settings=make_settings())
    fake = FakeAnthropic([FakeAnthropicResponse([FakeBlock("text", "ok")])])
    client._anthropic = fake  # type: ignore[assignment]

    await client.complete("prompt")

    assert fake.messages.calls[0]["system"] == llm_module.DEFAULT_SYSTEM_PROMPT


async def test_missing_anthropic_api_key_raises() -> None:
    client = LLMClient(settings=make_settings(anthropic_api_key=None))
    monkey = pytest.MonkeyPatch()
    monkey.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            _ = client.anthropic
    finally:
        monkey.undo()


async def test_missing_openai_api_key_raises() -> None:
    client = LLMClient(settings=make_settings(openai_api_key=None))
    monkey = pytest.MonkeyPatch()
    monkey.delenv("OPENAI_API_KEY", raising=False)
    try:
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            _ = client.openai
    finally:
        monkey.undo()


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #


async def test_openai_happy_path() -> None:
    client = LLMClient(settings=make_settings(provider=Provider.OPENAI))
    fake = FakeOpenAI([FakeOpenAIResponse("openai reply")])
    client._openai = fake  # type: ignore[assignment]

    assert await client.complete("hi", system="sys") == "openai reply"
    call = fake.chat.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["max_tokens"] == MAX_TOKENS
    assert call["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


async def test_openai_none_content_becomes_empty_string() -> None:
    client = LLMClient(settings=make_settings(provider=Provider.OPENAI))
    client._openai = FakeOpenAI([FakeOpenAIResponse(None)])  # type: ignore[assignment]

    assert await client.complete("hi") == ""


async def test_openai_forwards_explicit_model() -> None:
    client = LLMClient(settings=make_settings(provider=Provider.OPENAI))
    fake = FakeOpenAI([FakeOpenAIResponse("ok")])
    client._openai = fake  # type: ignore[assignment]

    await client.complete("hi", model="override-model")

    assert fake.chat.completions.calls[0]["model"] == "override-model"


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #


async def test_ollama_default_base_url(fake_ollama: type[FakeAsyncClientFactory]) -> None:
    client = LLMClient(settings=make_settings(provider=Provider.OLLAMA))

    result = await client.complete("hi")

    assert result == "ollama says hi"
    assert fake_ollama.requests[0][0] == "http://localhost:11434/api/chat"


async def test_ollama_trailing_slash_base_url(
    fake_ollama: type[FakeAsyncClientFactory],
) -> None:
    client = LLMClient(
        settings=make_settings(provider=Provider.OLLAMA, ollama_base_url="http://host:1234/")
    )

    await client.complete("hi")

    assert fake_ollama.requests[0][0] == "http://host:1234/api/chat"


async def test_ollama_subpath_base_url(fake_ollama: type[FakeAsyncClientFactory]) -> None:
    client = LLMClient(
        settings=make_settings(
            provider=Provider.OLLAMA, ollama_base_url="http://host:1234/proxy/ollama"
        )
    )

    await client.complete("hi")

    assert fake_ollama.requests[0][0] == "http://host:1234/proxy/ollama/api/chat"


async def test_ollama_payload_and_system_prompt(fake_ollama: type[FakeAsyncClientFactory]) -> None:
    client = LLMClient(settings=make_settings(provider=Provider.OLLAMA, model="llama3"))

    await client.complete("prompt text", system="be nice")

    url, payload = fake_ollama.requests[0]
    assert url.endswith("/api/chat")
    assert payload["model"] == "llama3"
    assert payload["stream"] is False
    assert payload["messages"] == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "prompt text"},
    ]


async def test_ollama_default_system_prompt(fake_ollama: type[FakeAsyncClientFactory]) -> None:
    client = LLMClient(settings=make_settings(provider=Provider.OLLAMA))

    await client.complete("prompt text")

    _, payload = fake_ollama.requests[0]
    assert payload["messages"][0]["content"] == llm_module.DEFAULT_SYSTEM_PROMPT


async def test_ollama_without_message_object_raises(
    fake_ollama: type[FakeAsyncClientFactory],
) -> None:
    client = LLMClient(settings=make_settings(provider=Provider.OLLAMA))
    fake_ollama.payload = {"unexpected": True}

    with pytest.raises(ValueError, match="no message object"):
        await client.complete("hi")


# --------------------------------------------------------------------------- #
# Provider dispatch / settings
# --------------------------------------------------------------------------- #


async def test_unknown_provider_raises(fake_ollama: type[FakeAsyncClientFactory]) -> None:
    settings = make_settings()
    object.__setattr__(settings, "provider", "not-a-provider")
    client = LLMClient(settings=settings)

    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hi")


def test_client_defaults_to_singleton_settings() -> None:
    assert LLMClient().settings is llm_module.default_settings
    assert LLMClient(settings=None).settings is llm_module.default_settings


def test_client_keeps_explicit_settings() -> None:
    settings = make_settings(provider=Provider.OPENAI)
    assert LLMClient(settings=settings).settings is settings


def test_client_construction_does_not_create_clients() -> None:
    client = LLMClient(settings=make_settings())
    assert client._anthropic is None
    assert client._openai is None


# --------------------------------------------------------------------------- #
# Retry behaviour
# --------------------------------------------------------------------------- #


async def test_retry_after_transient_transport_error(fast_retries: None) -> None:
    client = LLMClient(settings=make_settings())
    fake = FakeAnthropic(
        [
            httpx.ConnectError("connection reset"),
            httpx.ReadTimeout("timed out"),
            FakeAnthropicResponse([FakeBlock("text", "recovered")]),
        ]
    )
    client._anthropic = fake  # type: ignore[assignment]

    assert await client.complete("hi") == "recovered"
    assert len(fake.messages.calls) == 3


async def test_retry_after_server_error_status(fast_retries: None) -> None:
    client = LLMClient(settings=make_settings())
    fake = FakeAnthropic(
        [
            status_error(anthropic.APIStatusError, 503, "unavailable"),
            status_error(anthropic.APIStatusError, 500, "server error"),
            FakeAnthropicResponse([FakeBlock("text", "recovered")]),
        ]
    )
    client._anthropic = fake  # type: ignore[assignment]

    assert await client.complete("hi") == "recovered"
    assert len(fake.messages.calls) == 3


async def test_retry_after_rate_limit(fast_retries: None) -> None:
    client = LLMClient(settings=make_settings(provider=Provider.OPENAI))
    fake = FakeOpenAI(
        [
            status_error(anthropic.RateLimitError, 429, "slow down"),
            FakeOpenAIResponse("recovered"),
        ]
    )
    client._openai = fake  # type: ignore[assignment]

    assert await client.complete("hi") == "recovered"
    assert len(fake.chat.completions.calls) == 2


async def test_retry_exhausted_reraises(fast_retries: None) -> None:
    client = LLMClient(settings=make_settings())
    fake = FakeAnthropic([httpx.ConnectError("down")] * 5)
    client._anthropic = fake  # type: ignore[assignment]

    with pytest.raises(httpx.ConnectError):
        await client.complete("hi")
    assert len(fake.messages.calls) == 3


@pytest.mark.parametrize(
    "error",
    [
        status_error(anthropic.AuthenticationError, 401, "bad key"),
        status_error(anthropic.BadRequestError, 400, "bad request"),
        status_error(anthropic.APIStatusError, 404, "not found"),
        ValueError("nope"),
    ],
)
async def test_non_retryable_errors_raise_immediately(
    fast_retries: None, error: Exception
) -> None:
    client = LLMClient(settings=make_settings())
    fake = FakeAnthropic([error])
    client._anthropic = fake  # type: ignore[assignment]

    with pytest.raises(type(error)):
        await client.complete("hi")
    assert len(fake.messages.calls) == 1


async def test_cancelled_error_is_never_retried(fast_retries: None) -> None:
    import asyncio

    client = LLMClient(settings=make_settings())
    fake = FakeAnthropic([asyncio.CancelledError()])
    client._anthropic = fake  # type: ignore[assignment]

    with pytest.raises(asyncio.CancelledError):
        await client.complete("hi")
    assert len(fake.messages.calls) == 1


def test_transient_classification() -> None:
    predicate = llm_module._RetryTransientFailures()
    assert llm_module._is_transient_failure(httpx.ConnectError("x")) is True
    assert llm_module._is_transient_failure(httpx.ReadTimeout("x")) is True
    assert (
        llm_module._is_transient_failure(status_error(anthropic.APIStatusError, 502)) is True
    )
    assert llm_module._is_transient_failure(status_error(anthropic.RateLimitError, 429)) is True
    assert (
        llm_module._is_transient_failure(status_error(anthropic.AuthenticationError, 401))
        is False
    )
    assert llm_module._is_transient_failure(status_error(anthropic.BadRequestError, 400)) is False
    assert llm_module._is_transient_failure(RuntimeError("x")) is False
    assert llm_module._is_transient_failure(None) is False
    assert callable(predicate)


# --------------------------------------------------------------------------- #
# File helpers
# --------------------------------------------------------------------------- #


async def test_read_write_roundtrip(tmp_path: Path) -> None:
    client = LLMClient(settings=make_settings())
    target = tmp_path / "nested" / "dir" / "file.txt"

    await client.write_file(target, "hello world")

    assert target.read_text() == "hello world"
    assert await client.read_file(target) == "hello world"
    assert await client.read_file(str(target)) == "hello world"


async def test_read_file_returns_empty_for_empty_file(tmp_path: Path) -> None:
    client = LLMClient(settings=make_settings())
    target = tmp_path / "empty.txt"

    await client.write_file(target, "")

    assert await client.read_file(target) == ""


def test_list_files_excludes_noise_and_sorts(tmp_path: Path) -> None:
    files = [
        "zeta.py",
        "alpha.py",
        "pkg/module.py",
        "__pycache__/alpha.cpython-310.pyc",
        ".git/config",
        ".venv/lib/thing.py",
        "node_modules/pkg/index.js",
        ".hidden/secret.txt",
        "compiled.pyc",
        ".env",
    ]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")

    client = LLMClient(settings=make_settings())
    result = client.list_files(tmp_path)

    assert result == sorted(result)
    assert result == ["alpha.py", "pkg/module.py", "zeta.py"]
    assert all(not item.startswith(".") for item in result)


def test_list_files_missing_directory_returns_empty(tmp_path: Path) -> None:
    client = LLMClient(settings=make_settings())
    assert client.list_files(tmp_path / "does-not-exist") == []
    # tmp_path is empty, so list_files should return empty list
    assert client.list_files(str(tmp_path)) == []


def test_list_files_optional_kwargs(tmp_path: Path) -> None:
    for name in ("a.py", "b.py", "skip_me.py", "sub/skip_me.py", "sub/keep.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")

    client = LLMClient(settings=make_settings())

    assert client.list_files(tmp_path, exclude=["skip_me.py"]) == ["a.py", "b.py", "sub/keep.py"]
    assert client.list_files(tmp_path, exclude=["sub"]) == ["a.py", "b.py", "skip_me.py"]
    assert client.list_files(tmp_path, max_results=2) == ["a.py", "b.py"]
    assert client.list_files(tmp_path, max_results=0) == []
    assert client.list_files(tmp_path, exclude=(), max_results=None) == [
        "a.py",
        "b.py",
        "skip_me.py",
        "sub/keep.py",
        "sub/skip_me.py",
    ]
