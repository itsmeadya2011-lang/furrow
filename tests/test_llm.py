"""Regression tests for :mod:`furrow.llm`.

These tests never touch the network: every provider SDK client is replaced with a stub that
records its arguments and returns canned responses, and the Ollama branch is driven by an
``httpx.MockTransport``. The fixtures in ``conftest.py`` provide the hermetic ``settings``
(with ``anthropic_api_key="test-key"``) and a temp ``tmp_workspace``.
"""

from __future__ import annotations

from typing import cast

import httpx
import pytest

from furrow.config import Provider, Settings
from furrow.llm import (
    DEFAULT_SYSTEM,
    LLMClient,
    LLMError,
    LLMResponseTruncated,
    ProviderAuthError,
    ProviderUnavailableError,
)

# ---------------------------------------------------------------------------
# Fake provider SDK clients (mirror the scratch scripts in /tmp/kilo/scratch_llm)
# ---------------------------------------------------------------------------


class _Block:
    """Minimal stand-in for an Anthropic content block.

    Only ``type`` and ``text`` are set, which is exactly what :meth:`LLMClient._complete_anthropic`
    inspects: it filters by ``block.type == "text"`` and reads ``block.text``.
    """

    def __init__(self, type_: str, text: str = "") -> None:
        self.type = type_
        if text:
            self.text = text


class _AnthropicMessage:
    def __init__(self, content, stop_reason: str = "end_turn") -> None:
        self.content = content
        self.stop_reason = stop_reason


class _AnthropicMessages:
    def __init__(self, result=None, error=None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class _FakeAnthropic:
    def __init__(self, messages: _AnthropicMessages) -> None:
        self.messages = messages


class _Choice:
    def __init__(self, content, finish_reason: str = "stop") -> None:
        self.message = type("_M", (), {"content": content})()
        self.finish_reason = finish_reason


class _Completions:
    def __init__(self, choices=None, error=None) -> None:
        self.choices = choices
        self.error = error
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return type("_R", (), {"choices": self.choices})()


class _FakeOpenAI:
    def __init__(self, completions: _Completions) -> None:
        self.chat = type("_Chat", (), {"completions": completions})()


def _http_error(cls, status: int) -> Exception:
    """Build a real SDK status error carrying a synthetic httpx response."""
    return cls(
        f"status {status}",
        response=httpx.Response(status, request=httpx.Request("POST", "https://example.test")),
        body=None,
    )


def _anthropic_client(settings: Settings, result=None, error=None) -> LLMClient:
    client = LLMClient(settings)
    client._anthropic = _FakeAnthropic(_AnthropicMessages(result=result, error=error))  # type: ignore[assignment]
    return client


def _openai_client(settings: Settings, choices=None, error=None) -> tuple[LLMClient, _Completions]:
    completions = _Completions(choices=choices, error=error)
    client = LLMClient(settings)
    client._openai = _FakeOpenAI(completions)  # type: ignore[assignment]
    return client, completions


def _zero_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make :meth:`LLMClient.complete` retry without sleeping.

    ``tenacity`` decorates the coroutine on the class, so the wait strategy is an attribute of
    the ``AsyncRetrying`` object that ``LLMClient.complete.retry`` exposes. Patching it in place
    keeps the retry count and backoff logic intact while shaving the seconds off every retry
    test. This is the one private attribute touched; everything else is exercised through the
    public API.
    """
    import tenacity

    # Monkeypatching a private attribute of the tenacity wrapper around `complete`.
    monkeypatch.setattr(
        LLMClient.complete.retry,  # type: ignore[attr-defined]
        "wait",
        tenacity.wait_fixed(0),
    )


#: The retry budget configured on :meth:`LLMClient.complete` (``stop_after_attempt(4)``).
#:
#: Retry tests assert against this exact number, so the policy is pinned rather than merely
#: "retried at least once", and the exact attempt count is still checked while the backoff
#: itself is short-circuited by :func:`_zero_retry_wait`.
_RETRY_ATTEMPTS = 4


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_argument_stores_real_settings(settings: Settings) -> None:
    """``LLMClient()`` and ``LLMClient(None)`` resolve to the process settings, not ``None``.

    The old signature was ``def __init__(self, settings: Settings = settings)``: the default
    *argument* was the module-level ``settings`` singleton, but ``None`` was stored verbatim and
    the first attribute access inside a provider property blew up with ``AttributeError``.
    """
    default_client = LLMClient()
    none_client = LLMClient(None)

    assert isinstance(default_client.settings, Settings)
    assert isinstance(none_client.settings, Settings)
    # Both resolve to the same cached singleton, so they are interchangeable.
    assert default_client.settings is none_client.settings
    # The stored object is a real Settings, so attribute access does not raise (the old
    # `settings=None` default stored None and crashed on first attribute access).
    _ = default_client.settings.provider
    _ = none_client.settings.max_tokens


def test_explicit_settings_is_stored_verbatim() -> None:
    custom = Settings(provider=Provider.OPENAI, openai_api_key="k", max_tokens=42)
    client = LLMClient(custom)
    assert client.settings is custom
    assert client.settings.max_tokens == 42


# ---------------------------------------------------------------------------
# Provider properties
# ---------------------------------------------------------------------------


def test_anthropic_property_raises_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FURROW_ANTHROPIC_API_KEY", raising=False)
    client = LLMClient(Settings(provider=Provider.ANTHROPIC))
    with pytest.raises(ProviderAuthError) as exc_info:
        _ = client.anthropic
    assert "ANTHROPIC_API_KEY" in str(exc_info.value)


def test_openai_property_raises_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FURROW_OPENAI_API_KEY", raising=False)
    client = LLMClient(Settings(provider=Provider.OPENAI))
    with pytest.raises(ProviderAuthError) as exc_info:
        _ = client.openai
    assert "OPENAI_API_KEY" in str(exc_info.value)


def test_anthropic_property_is_cached_and_lazy(settings: Settings) -> None:
    """The cached client is built once and the construction makes no network call."""
    from anthropic import AsyncAnthropic

    client = LLMClient(settings)
    first = client.anthropic
    second = client.anthropic
    assert first is second
    assert isinstance(first, AsyncAnthropic)


def test_openai_property_is_cached_and_lazy(settings: Settings) -> None:
    from openai import AsyncOpenAI

    client = LLMClient(settings)
    first = client.openai
    second = client.openai
    assert first is second
    assert isinstance(first, AsyncOpenAI)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


async def test_unknown_provider_raises_llm_error() -> None:
    client = LLMClient(Settings(provider=Provider.ANTHROPIC, anthropic_api_key="k"))
    # Provider is a str Enum; assigning a bogus value bypasses the enum and exercises the
    # fallthrough branch that the old code never reached.
    client.settings.provider = "bogus"  # type: ignore[assignment]
    with pytest.raises(LLMError) as exc_info:
        await client.complete("hi")
    assert "bogus" in str(exc_info.value)


def test_max_tokens_defaults_and_can_be_overridden() -> None:
    settings = Settings(provider=Provider.OPENAI, openai_api_key="k")
    client, completions = _openai_client(settings, choices=[_Choice("ok")])

    async def run() -> None:
        await client.complete("hi")
        assert completions.calls[0]["max_tokens"] == settings.max_tokens
        await client.complete("hi", max_tokens=16)
        assert completions.calls[1]["max_tokens"] == 16

    import asyncio

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Anthropic response handling
# ---------------------------------------------------------------------------


async def test_anthropic_thinking_block_first_returns_text_block(settings: Settings) -> None:
    """A thinking block precedes the text block; the TEXT block's content is returned.

    The old code did ``response.content[0].text``, which is the thinking block and has no
    ``.text`` attribute at all.
    """
    client = _anthropic_client(
        settings,
        result=_AnthropicMessage(
            [_Block("thinking"), _Block("text", "hello "), _Block("text", "world")]
        ),
    )
    text = await client.complete("hi", system="be terse")
    assert text == "hello world"


async def test_anthropic_system_is_forwarded(settings: Settings) -> None:
    """`system` is sent as the real ``system`` parameter (the old code ignored it)."""
    client = _anthropic_client(settings, result=_AnthropicMessage([_Block("text", "ok")]))
    await client.complete("hi", system="you are a pirate")
    messages = cast(_FakeAnthropic, client._anthropic).messages
    payload = messages.calls[0]
    assert payload["system"] == "you are a pirate"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


async def test_anthropic_empty_system_uses_default(settings: Settings) -> None:
    client = _anthropic_client(settings, result=_AnthropicMessage([_Block("text", "ok")]))
    await client.complete("hi")
    messages = cast(_FakeAnthropic, client._anthropic).messages
    assert messages.calls[0]["system"] == DEFAULT_SYSTEM


async def test_anthropic_no_text_block_raises_naming_block_types(settings: Settings) -> None:
    client = _anthropic_client(
        settings, result=_AnthropicMessage([_Block("thinking"), _Block("tool_use")])
    )
    with pytest.raises(LLMError) as exc_info:
        await client.complete("hi")
    message = str(exc_info.value)
    assert "thinking" in message
    assert "tool_use" in message


async def test_anthropic_max_tokens_raises_truncated(settings: Settings) -> None:
    client = _anthropic_client(
        settings,
        result=_AnthropicMessage([_Block("text", "part")], stop_reason="max_tokens"),
    )
    with pytest.raises(LLMResponseTruncated):
        await client.complete("hi")


async def test_anthropic_auth_error_maps_to_provider_auth_error(settings: Settings) -> None:
    import anthropic

    client = _anthropic_client(
        settings, error=_http_error(anthropic.AuthenticationError, 401)
    )
    with pytest.raises(ProviderAuthError):
        await client.complete("hi")


async def test_anthropic_rate_limit_maps_to_unavailable(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 429 is retried up to the budget, then surfaces as ProviderUnavailableError."""
    _zero_retry_wait(monkeypatch)
    import anthropic

    messages = _AnthropicMessages(error=_http_error(anthropic.RateLimitError, 429))
    client = LLMClient(settings)
    client._anthropic = _FakeAnthropic(messages)  # type: ignore[assignment]
    with pytest.raises(ProviderUnavailableError):
        await client.complete("hi")
    assert len(messages.calls) == _RETRY_ATTEMPTS


async def test_anthropic_5xx_maps_to_unavailable(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 503 is retried up to the budget, then surfaces as ProviderUnavailableError."""
    _zero_retry_wait(monkeypatch)
    import anthropic

    messages = _AnthropicMessages(error=_http_error(anthropic.InternalServerError, 503))
    client = LLMClient(settings)
    client._anthropic = _FakeAnthropic(messages)  # type: ignore[assignment]
    with pytest.raises(ProviderUnavailableError):
        await client.complete("hi")
    assert len(messages.calls) == _RETRY_ATTEMPTS


async def test_anthropic_4xx_maps_to_llm_error(settings: Settings) -> None:
    import anthropic

    client = _anthropic_client(
        settings, error=_http_error(anthropic.BadRequestError, 400)
    )
    with pytest.raises(LLMError) as exc_info:
        await client.complete("hi")
    assert "400" in str(exc_info.value)


# ---------------------------------------------------------------------------
# OpenAI response handling
# ---------------------------------------------------------------------------


async def test_openai_finish_reason_length_raises_truncated() -> None:
    client, _ = _openai_client(
        _openai_settings(), choices=[_Choice("partial", finish_reason="length")]
    )
    with pytest.raises(LLMResponseTruncated):
        await client.complete("hi")


async def test_openai_system_omitted_when_empty() -> None:
    client, completions = _openai_client(_openai_settings(), choices=[_Choice("ok")])
    await client.complete("hi")
    assert completions.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


async def test_openai_system_present_when_nonempty() -> None:
    client, completions = _openai_client(_openai_settings(), choices=[_Choice("ok")])
    await client.complete("hi", system="be terse")
    assert completions.calls[0]["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


async def test_openai_empty_content_raises() -> None:
    client, _ = _openai_client(_openai_settings(), choices=[_Choice(None, finish_reason="stop")])
    with pytest.raises(LLMError):
        await client.complete("hi")


async def test_openai_no_choices_raises() -> None:
    client, _ = _openai_client(_openai_settings(), choices=[])
    with pytest.raises(LLMError):
        await client.complete("hi")


async def test_openai_auth_error_maps() -> None:
    import openai

    client, _ = _openai_client(
        _openai_settings(), error=_http_error(openai.AuthenticationError, 401)
    )
    with pytest.raises(ProviderAuthError):
        await client.complete("hi")


async def test_openai_rate_limit_maps(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 is retried up to the budget, then surfaces as ProviderUnavailableError."""
    _zero_retry_wait(monkeypatch)
    import openai

    client, completions = _openai_client(
        _openai_settings(), error=_http_error(openai.RateLimitError, 429)
    )
    with pytest.raises(ProviderUnavailableError):
        await client.complete("hi")
    assert len(completions.calls) == _RETRY_ATTEMPTS


async def test_openai_4xx_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 4xx client error is not retried: the call count is exactly 1."""
    _zero_retry_wait(monkeypatch)
    import openai

    class BadRequest(_Completions):
        async def create(self, **kwargs):
            self.calls.append(kwargs)
            raise _http_error(openai.BadRequestError, 400)

    completions = BadRequest()
    client = LLMClient(_openai_settings())
    client._openai = _FakeOpenAI(completions)  # type: ignore[assignment]
    with pytest.raises(LLMError):
        await client.complete("hi")
    assert len(completions.calls) == 1


async def test_openai_transient_failure_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient failure is retried and then succeeds; the call count is greater than 1."""
    _zero_retry_wait(monkeypatch)
    import openai

    class Flaky(_Completions):
        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) < 3:
                raise _http_error(openai.RateLimitError, 429)
            return type("_R", (), {"choices": [_Choice("recovered")]})()

    completions = Flaky()
    client = LLMClient(_openai_settings())
    client._openai = _FakeOpenAI(completions)  # type: ignore[assignment]
    text = await client.complete("hi")
    assert text == "recovered"
    assert len(completions.calls) == 3


# ---------------------------------------------------------------------------
# Ollama branch
# ---------------------------------------------------------------------------


def _ollama_settings(**overrides) -> Settings:
    return Settings(
        provider=Provider.OLLAMA,
        ollama_base_url="http://ollama.test:11434",
        ollama_model="llama3.1",
        **overrides,
    )


def _openai_settings(**overrides) -> Settings:
    """OPENAI provider with a dummy key; the fixture is ANTHROPIC, so OpenAI tests need this."""
    return Settings(provider=Provider.OPENAI, openai_api_key="k", **overrides)


def _patch_httpx(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Swap ``httpx.AsyncClient`` for one that routes through a ``MockTransport``.

    ``LLMClient._complete_ollama`` builds its own ``httpx.AsyncClient`` per request, so the
    constructor is the single seam; patching the class is the least invasive way to steer it.
    """
    base = httpx.AsyncClient

    class _Client(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)


async def test_ollama_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append({"path": request.url.path, "payload": request.read().decode()})
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "hi!"}},
        )

    _patch_httpx(monkeypatch, handler)
    client = LLMClient(_ollama_settings())
    text = await client.complete("hello", system="be terse")
    assert text == "hi!"
    import json

    payload = json.loads(captured[0]["payload"])
    assert payload["model"] == "llama3.1"
    assert payload["stream"] is False
    assert payload["options"]["num_predict"] == 8192
    assert payload["messages"][0] == {"role": "system", "content": "be terse"}
    assert payload["messages"][1] == {"role": "user", "content": "hello"}


async def test_ollama_empty_system_omits_system_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """When `system` is empty the Ollama request carries a user message only.

    Consistent with the OpenAI branch, an empty system prompt is not padded out with a filler.
    """
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.read().decode())
        return httpx.Response(200, json={"message": {"content": "ok"}})

    _patch_httpx(monkeypatch, handler)
    client = LLMClient(_ollama_settings())
    await client.complete("hello")
    import json

    assert json.loads(seen[0])["messages"] == [{"role": "user", "content": "hello"}]


async def test_ollama_nonempty_system_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.read().decode())
        return httpx.Response(200, json={"message": {"content": "ok"}})

    _patch_httpx(monkeypatch, handler)
    client = LLMClient(_ollama_settings())
    await client.complete("hello", system="be terse")
    import json

    assert json.loads(seen[0])["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"},
    ]


async def test_ollama_connection_failure_raises_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreachable daemon is retried up to the budget, then surfaces as unavailable."""
    _zero_retry_wait(monkeypatch)
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise httpx.ConnectError("refused")

    _patch_httpx(monkeypatch, handler)
    client = LLMClient(_ollama_settings())
    with pytest.raises(ProviderUnavailableError):
        await client.complete("hello")
    assert len(attempts) == _RETRY_ATTEMPTS


async def test_ollama_5xx_raises_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 503 from the daemon is retried up to the budget, then surfaces as unavailable."""
    _zero_retry_wait(monkeypatch)
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(503, text="overloaded")

    _patch_httpx(monkeypatch, handler)
    client = LLMClient(_ollama_settings())
    with pytest.raises(ProviderUnavailableError):
        await client.complete("hello")
    assert len(attempts) == _RETRY_ATTEMPTS


async def test_ollama_4xx_raises_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    _patch_httpx(monkeypatch, handler)
    client = LLMClient(_ollama_settings())
    with pytest.raises(LLMError):
        await client.complete("hello")


async def test_ollama_done_reason_length_raises_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"done": True, "done_reason": "length", "message": {"content": "part"}},
        )

    _patch_httpx(monkeypatch, handler)
    client = LLMClient(_ollama_settings())
    with pytest.raises(LLMResponseTruncated):
        await client.complete("hello")


async def test_ollama_empty_content_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"done": True, "message": {"content": ""}})

    _patch_httpx(monkeypatch, handler)
    client = LLMClient(_ollama_settings())
    with pytest.raises(LLMError):
        await client.complete("hello")


async def test_ollama_non_json_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    _patch_httpx(monkeypatch, handler)
    client = LLMClient(_ollama_settings())
    with pytest.raises(LLMError):
        await client.complete("hello")


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


async def test_read_file_returns_contents(tmp_workspace) -> None:
    client = LLMClient()
    target = tmp_workspace / "file.txt"
    target.write_text("hello world", encoding="utf-8")
    try:
        assert await client.read_file(target) == "hello world"
    finally:
        await client.aclose()


async def test_read_file_missing_raises_oserror(tmp_workspace) -> None:
    client = LLMClient()
    try:
        with pytest.raises(OSError):
            await client.read_file(tmp_workspace / "missing.txt")
    finally:
        await client.aclose()


async def test_read_file_truncates_with_marker(tmp_workspace) -> None:
    client = LLMClient()
    target = tmp_workspace / "big.txt"
    target.write_bytes(b"x" * 5000)
    try:
        result = await client.read_file(target, max_bytes=100)
        assert result.startswith("x" * 100)
        assert "truncated" in result
        assert "4900 of 5000" in result
    finally:
        await client.aclose()


async def test_read_file_handles_non_utf8(tmp_workspace) -> None:
    client = LLMClient()
    target = tmp_workspace / "bin.dat"
    target.write_bytes(b"\xff\xfe\x00not utf-8")
    try:
        result = await client.read_file(target)
        assert isinstance(result, str)
    finally:
        await client.aclose()


async def test_write_file_creates_parents_and_returns_path(tmp_workspace) -> None:
    client = LLMClient()
    try:
        target = tmp_workspace / "a" / "b" / "c.txt"
        result = await client.write_file(target, "héllo wörld\n")
        assert result == target
        assert target.read_text(encoding="utf-8") == "héllo wörld\n"
    finally:
        await client.aclose()


def test_list_files_recursive_and_pattern(tmp_workspace) -> None:
    client = LLMClient()
    (tmp_workspace / "pkg").mkdir()
    (tmp_workspace / "pkg" / "deep").mkdir()
    (tmp_workspace / "pkg" / "deep" / "mod.py").write_text("# x")
    (tmp_workspace / "top.py").write_text("# y")
    (tmp_workspace / "notes.md").write_text("# z")

    py_files = client.list_files(tmp_workspace, pattern="*.py")
    assert "top.py" in py_files
    assert "pkg/deep/mod.py" in py_files
    assert "notes.md" not in py_files


def test_list_files_skips_reserved_directories(tmp_workspace) -> None:
    client = LLMClient()
    for name in ("__pycache__", ".git", "node_modules", ".venv", "venv"):
        (tmp_workspace / name).mkdir()
        (tmp_workspace / name / "leaked.py").write_text("# nope")
    (tmp_workspace / "real.py").write_text("# yes")
    assert client.list_files(tmp_workspace, pattern="*.py") == ["real.py"]


def test_list_files_skip_hidden(tmp_workspace) -> None:
    client = LLMClient()
    (tmp_workspace / ".hidden.py").write_text("# dotfile")
    (tmp_workspace / "visible.py").write_text("# ok")
    assert client.list_files(tmp_workspace, pattern="*.py") == ["visible.py"]
    assert client.list_files(tmp_workspace, pattern="*.py", skip_hidden=False) == [
        ".hidden.py",
        "visible.py",
    ]


def test_list_files_max_entries_caps(tmp_workspace) -> None:
    client = LLMClient()
    for i in range(5):
        (tmp_workspace / f"mod{i}.py").write_text("# filler")
    capped = client.list_files(tmp_workspace, pattern="*.py", max_entries=2)
    assert len(capped) == 2
    assert capped == ["mod0.py", "mod1.py"]


def test_list_files_missing_directory_returns_empty(tmp_workspace) -> None:
    client = LLMClient()
    assert client.list_files(tmp_workspace / "nope") == []


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_aclose_is_idempotent(settings: Settings) -> None:
    client = LLMClient(settings)
    _ = client.anthropic  # build the cached client
    await client.aclose()
    assert client._anthropic is None
    # Calling again must not raise.
    await client.aclose()


async def test_async_context_manager_closes_clients(settings: Settings) -> None:
    async with LLMClient(settings) as client:
        assert isinstance(client, LLMClient)
        _ = client.anthropic
        assert client._anthropic is not None
    assert client._anthropic is None
