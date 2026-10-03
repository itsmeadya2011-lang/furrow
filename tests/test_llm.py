"""Offline tests for :mod:`furrow.llm`.

Everything here is stubbed: no network access, no API keys, no real SDK clients.
"""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import httpx
import openai
import pytest
from tenacity import wait_none

from furrow import llm as llm_module
from furrow.config import Provider, Settings
from furrow.llm import RETRY_KWARGS, TRANSIENT_ERRORS, LLMClient, is_transient_error


# --------------------------------------------------------------------------
# helpers / stubs (all local to this file)
# --------------------------------------------------------------------------
def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.example.invalid/v1/messages")


def _response(status: int = 429) -> httpx.Response:
    return httpx.Response(status, request=_request())


def make_settings(**kwargs: object) -> Settings:
    base: dict[str, object] = {"anthropic_api_key": "k", "openai_api_key": "k"}
    base.update(kwargs)
    return Settings(**base)  # type: ignore[arg-type]


class StubMessages:
    """Echoes the request kwargs and replies with ``content`` blocks."""

    def __init__(self, recorder: list[dict], content: list[object]) -> None:
        self._recorder = recorder
        self._content = content

    async def create(self, **kwargs: object) -> object:
        self._recorder.append(dict(kwargs))
        return SimpleNamespace(content=list(self._content))


class StubAnthropic:
    def __init__(self, recorder: list[dict], content: list[object]) -> None:
        self.messages = StubMessages(recorder, content)


class StubCompletions:
    def __init__(self, recorder: list[dict]) -> None:
        self._recorder = recorder

    async def create(self, **kwargs: object) -> object:
        self._recorder.append(dict(kwargs))

        class Message:
            content = "openai-reply"

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()


class StubOpenAI:
    def __init__(self, recorder: list[dict]) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = StubCompletions(recorder)


class StubResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.raised = False

    def raise_for_status(self) -> None:
        self.raised = True
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=_request(),
                response=httpx.Response(self.status_code, request=_request()),
            )

    def json(self) -> object:
        return self._payload


class StubAsyncClient:
    """Records constructor kwargs and each post so tests can assert payloads."""

    instances: list[StubAsyncClient] = []
    response_payload: object = {"message": {"content": "ollama-reply"}}
    status_code: int = 200

    def __init__(self, **kwargs: object) -> None:
        self.init_kwargs = dict(kwargs)
        self.posts: list[tuple[str, dict]] = []
        StubAsyncClient.instances.append(self)

    async def __aenter__(self) -> StubAsyncClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def post(self, url: str, **kwargs: object) -> StubResponse:
        self.posts.append((url, dict(kwargs)))
        return StubResponse(StubAsyncClient.response_payload, StubAsyncClient.status_code)


@pytest.fixture
def stub_httpx(monkeypatch: pytest.MonkeyPatch) -> type[StubAsyncClient]:
    StubAsyncClient.instances = []
    StubAsyncClient.response_payload = {"message": {"content": "ollama-reply"}}
    StubAsyncClient.status_code = 200
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", StubAsyncClient)
    return StubAsyncClient


# --------------------------------------------------------------------------
# (a) ollama dispatch
# --------------------------------------------------------------------------
async def test_complete_dispatches_to_ollama(stub_httpx: type[StubAsyncClient]) -> None:
    client = LLMClient(make_settings(provider=Provider.OLLAMA, ollama_base_url="http://test.invalid"))

    result = await client.complete("hello there")

    assert result == "ollama-reply"
    assert len(stub_httpx.instances) == 1
    instance = stub_httpx.instances[0]
    assert instance.init_kwargs["base_url"] == "http://test.invalid"
    assert "timeout" in instance.init_kwargs
    assert len(instance.posts) == 1
    url, kwargs = instance.posts[0]
    assert url == "/api/chat"
    payload = kwargs["json"]
    assert payload["model"] == client.settings.model
    assert payload["stream"] is False
    assert payload["messages"][0] == {
        "role": "system",
        "content": "You are a helpful coding assistant.",
    }
    assert payload["messages"][1] == {"role": "user", "content": "hello there"}


async def test_complete_ollama_uses_custom_system_and_model(
    stub_httpx: type[StubAsyncClient],
) -> None:
    client = LLMClient(make_settings(provider=Provider.OLLAMA))

    await client.complete("p", system="be terse", model="llama3")

    payload = stub_httpx.instances[0].posts[0][1]["json"]
    assert payload["model"] == "llama3"
    assert payload["messages"][0]["content"] == "be terse"


async def test_complete_ollama_raises_on_missing_content(stub_httpx: type[StubAsyncClient]) -> None:
    stub_httpx.response_payload = {"message": {"role": "assistant"}}
    client = LLMClient(make_settings(provider=Provider.OLLAMA))

    with pytest.raises(ValueError, match="message.content"):
        await client.complete("hi")
    assert len(stub_httpx.instances) == 1


async def test_complete_ollama_raises_when_message_absent(
    stub_httpx: type[StubAsyncClient],
) -> None:
    stub_httpx.response_payload = {"error": "model not found"}
    client = LLMClient(make_settings(provider=Provider.OLLAMA))

    with pytest.raises(ValueError, match="message.content"):
        await client.complete("hi")


async def test_complete_ollama_surfaces_http_error(stub_httpx: type[StubAsyncClient]) -> None:
    stub_httpx.status_code = 500
    stub_httpx.response_payload = {"error": "server exploded"}
    client = LLMClient(make_settings(provider=Provider.OLLAMA))

    with pytest.raises(httpx.HTTPStatusError):
        await client.complete("hi")

    assert len(stub_httpx.instances) == 1
    # HTTPStatusError is not a TransportError, so it must not be retried.
    assert len(stub_httpx.instances[0].posts) == 1


# --------------------------------------------------------------------------
# (b) anthropic / openai dispatch
# --------------------------------------------------------------------------
def _record_dispatch(monkeypatch: pytest.MonkeyPatch, name: str, reply: str) -> list[dict]:
    """Patch one ``_complete_*`` method and record its invocation."""
    calls: list[dict] = []

    async def stub(self: LLMClient, prompt: str, system: str, model: str) -> str:
        calls.append({"prompt": prompt, "system": system, "model": model})
        return reply

    monkeypatch.setattr(LLMClient, name, stub)
    return calls


async def test_complete_dispatches_to_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_dispatch(monkeypatch, "_complete_anthropic", "anthropic-reply")
    client = LLMClient(make_settings(provider=Provider.ANTHROPIC))

    result = await client.complete("hello", model="claude-test")

    assert result == "anthropic-reply"
    assert calls == [{"prompt": "hello", "system": "", "model": "claude-test"}]


async def test_complete_ollama_is_not_used_for_other_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _record_dispatch(monkeypatch, "_complete_anthropic", "anthropic-reply")
    client = LLMClient(make_settings(provider=Provider.ANTHROPIC))

    await client.complete("hello")

    assert len(calls) == 1
    assert client.settings.provider is Provider.ANTHROPIC


async def test_complete_anthropic_body_uses_client(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    client = LLMClient(make_settings(provider=Provider.ANTHROPIC))
    monkeypatch.setattr(client, "_anthropic", StubAnthropic(calls, [SimpleNamespace(text="hello")]))

    result = await client._complete_anthropic("p", "sys", "claude-test")

    assert result == "hello"
    assert calls[0]["model"] == "claude-test"
    assert calls[0]["system"] == "sys"
    assert calls[0]["messages"] == [{"role": "user", "content": "p"}]


async def test_complete_anthropic_skips_blocks_without_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leading thinking block must not hide the text block behind it."""
    calls: list[dict] = []
    blocks = [SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="hello")]
    client = LLMClient(make_settings(provider=Provider.ANTHROPIC))
    monkeypatch.setattr(client, "_anthropic", StubAnthropic(calls, blocks))

    result = await client._complete_anthropic("p", "sys", "claude-test")

    assert result.strip() == "hello"
    assert calls[0]["messages"] == [{"role": "user", "content": "p"}]


async def test_complete_dispatches_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_dispatch(monkeypatch, "_complete_openai", "openai-reply")
    client = LLMClient(make_settings(provider=Provider.OPENAI))

    result = await client.complete("hello", model="gpt-test")

    assert result == "openai-reply"
    assert calls == [{"prompt": "hello", "system": "", "model": "gpt-test"}]


async def test_complete_openai_body_uses_client(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    client = LLMClient(make_settings(provider=Provider.OPENAI))
    monkeypatch.setattr(client, "_openai", StubOpenAI(calls))

    result = await client._complete_openai("p", "", "gpt-test")

    assert result == "openai-reply"
    assert calls[0]["model"] == "gpt-test"
    assert calls[0]["messages"][0] == {
        "role": "system",
        "content": "You are a helpful coding assistant.",
    }
    assert calls[0]["messages"][1] == {"role": "user", "content": "p"}


async def test_unknown_provider_raises() -> None:
    settings_obj = make_settings()
    object.__setattr__(settings_obj, "provider", "mystery-vendor")
    client = LLMClient(settings_obj)

    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hi")


# --------------------------------------------------------------------------
# (c) file helpers
# --------------------------------------------------------------------------
async def test_write_then_read_round_trip(tmp_path) -> None:
    client = LLMClient(make_settings())
    target = tmp_path / "a" / "b" / "note.txt"

    await client.write_file(target, "hello file\n")

    assert (tmp_path / "a" / "b").is_dir()
    assert await client.read_file(target) == "hello file\n"


async def test_write_file_creates_parent_directories(tmp_path) -> None:
    client = LLMClient(make_settings())
    target = tmp_path / "deep" / "nested" / "path" / "file.py"

    await client.write_file(target, "x = 1\n")

    assert target.read_text() == "x = 1\n"


async def test_write_file_accepts_str_path(tmp_path) -> None:
    client = LLMClient(make_settings())
    target = tmp_path / "str_path.txt"

    await client.write_file(str(target), "content")

    assert target.read_text() == "content"


# --------------------------------------------------------------------------
# (d) list_files
# --------------------------------------------------------------------------
def test_list_files_returns_relative_paths(tmp_path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("print('x')\n")
    (tmp_path / "root.txt").write_text("root\n")
    client = LLMClient(make_settings())

    files = sorted(client.list_files(tmp_path))

    assert files == ["pkg/mod.py", "root.txt"]
    assert all(not f.startswith(str(tmp_path)) for f in files)


def test_list_files_missing_directory_returns_empty(tmp_path) -> None:
    client = LLMClient(make_settings())

    assert client.list_files(tmp_path / "does-not-exist") == []


# --------------------------------------------------------------------------
# (e) transient classification + retry
# --------------------------------------------------------------------------
def test_transient_errors_includes_httpx_transport() -> None:
    # TRANSIENT_ERRORS is a flat tuple of classes, so membership is identity, not
    # subclassing. ConnectError is covered because it subclasses TransportError,
    # which is what is_transient_error matches on.
    assert httpx.TransportError in TRANSIENT_ERRORS
    assert issubclass(httpx.ConnectError, httpx.TransportError)
    assert is_transient_error(httpx.ConnectError("cannot connect"))


def test_non_transient_errors_are_not_retried() -> None:
    bad_request = anthropic.BadRequestError("bad", response=_response(400), body=None)
    auth_error = anthropic.AuthenticationError("nope", response=_response(401), body=None)
    assert not is_transient_error(bad_request)
    assert not is_transient_error(auth_error)
    assert not is_transient_error(
        openai.BadRequestError("bad", response=_response(400), body=None)
    )
    assert not is_transient_error(
        openai.AuthenticationError("nope", response=_response(401), body=None)
    )
    assert not is_transient_error(ValueError("Unsupported provider: nope"))
    assert not is_transient_error(RuntimeError("boom"))


def test_transient_errors_are_retried() -> None:
    assert is_transient_error(httpx.ConnectError("cannot connect"))
    assert is_transient_error(
        anthropic.RateLimitError("slow down", response=_response(429), body=None)
    )
    assert is_transient_error(
        openai.RateLimitError("slow down", response=_response(429), body=None)
    )
    assert is_transient_error(anthropic.APIConnectionError(request=_request()))
    assert is_transient_error(openai.APITimeoutError(request=_request()))


def test_retry_kwargs_shape() -> None:
    assert set(RETRY_KWARGS) == {"stop", "wait"}


async def test_complete_retries_transient_error_three_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"n": 0}

    async def always_rate_limited(prompt: str, system: str, model: str) -> str:
        attempts["n"] += 1
        raise anthropic.RateLimitError("slow down", response=_response(429), body=None)

    # wait_none() keeps the test instant; production RETRY_KWARGS still backs off.
    monkeypatch.setitem(RETRY_KWARGS, "wait", wait_none())

    client = LLMClient(make_settings(provider=Provider.ANTHROPIC))
    monkeypatch.setattr(client, "_complete_once", always_rate_limited)

    with pytest.raises(anthropic.RateLimitError):
        await client.complete("hi")

    assert attempts["n"] == 3


async def test_complete_does_not_retry_permanent_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}

    async def always_bad_request(prompt: str, system: str, model: str) -> str:
        attempts["n"] += 1
        raise anthropic.BadRequestError("nope", response=_response(400), body=None)

    monkeypatch.setitem(RETRY_KWARGS, "wait", wait_none())

    client = LLMClient(make_settings(provider=Provider.ANTHROPIC))
    monkeypatch.setattr(client, "_complete_once", always_bad_request)

    with pytest.raises(anthropic.BadRequestError):
        await client.complete("hi")

    assert attempts["n"] == 1


async def test_complete_recovers_after_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}

    async def flaky(prompt: str, system: str, model: str) -> str:
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise httpx.ConnectError("reset by peer")
        return "recovered"

    monkeypatch.setitem(RETRY_KWARGS, "wait", wait_none())

    client = LLMClient(make_settings(provider=Provider.ANTHROPIC))
    monkeypatch.setattr(client, "_complete_once", flaky)

    assert await client.complete("hi") == "recovered"
    assert attempts["n"] == 2