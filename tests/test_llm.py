"""Regression tests for :mod:`furrow.llm`.

Every provider interaction goes through an in-process fake (an injected SDK stand-in
or ``httpx.MockTransport``); the autouse ``no_real_network`` fixture in
``conftest.py`` fails the test if anything tries to open a real socket.

Load-bearing regressions pinned here:

* ``LLMClient(settings=None)`` must fall back to the module level settings
  singleton - this was the bug that broke every agent.
* ``extract_json`` copes with fenced, prose-wrapped and brace-laden output.
* ``complete()`` retries transient failures and never retries an auth error.
* Anthropic parsing joins every text block instead of reading ``content[0]``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from furrow.config import settings as global_settings
from furrow.llm import (
    SKIP_DIRECTORIES,
    LLMClient,
    extract_json_object,
    is_retryable_error,
)
from tests.conftest import make_settings

# ---------------------------------------------------------------------- fakes


class FakeBlock:
    """One Anthropic content block."""

    def __init__(self, block_type: str, text: str | None = None) -> None:
        self.type = block_type
        if text is not None:
            self.text = text


class FakeAnthropicResponse:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class FakeAnthropicMessages:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.kwargs: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeAnthropic:
    def __init__(self, outcomes: list[Any]) -> None:
        self.messages = FakeAnthropicMessages(outcomes)


class FakeOpenAIResponse:
    def __init__(self, content: str | None) -> None:
        message = type("Message", (), {"content": content})()
        self.choices = [type("Choice", (), {"message": message})()]


class FakeOpenAICompletions:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.kwargs: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeOpenAI:
    def __init__(self, outcomes: list[Any]) -> None:
        self.completions = FakeOpenAICompletions(outcomes)
        self.chat = type("Chat", (), {"completions": self.completions})()


class AuthenticationError(Exception):
    """Stands in for ``anthropic.AuthenticationError``."""


class RateLimitError(Exception):
    """Stands in for ``anthropic.RateLimitError``."""


class StatusError(Exception):
    """An SDK error carrying an HTTP status code."""

    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(message or f"status {status}")
        self.status_code = status


def ollama_client(handler: Any, tmp_path: Path, **overrides: Any) -> LLMClient:
    """An :class:`LLMClient` whose HTTP calls are served by *handler*."""
    client = LLMClient(settings=make_settings(tmp_path, provider="ollama", **overrides))
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


# ------------------------------------------------------------ settings lookup


def test_client_without_settings_falls_back_to_the_module_singleton() -> None:
    """The regression: ``LLMClient()`` must not end up with unusable settings."""
    client = LLMClient()
    assert client.settings is global_settings


def test_client_with_settings_uses_them(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_cycles=4)
    assert LLMClient(settings=settings).settings is settings


def test_client_explicitly_passing_the_singleton_keeps_it() -> None:
    assert LLMClient(settings=global_settings).settings is global_settings


def test_agents_reach_the_singleton_through_a_default_client() -> None:
    """``PlannerAgent(client=None)`` builds ``LLMClient(settings=None)``."""
    from furrow.agents.planner import PlannerAgent

    agent = PlannerAgent()
    assert isinstance(agent.client, LLMClient)
    assert agent.client.settings is global_settings


def test_missing_api_key_is_reported_before_any_client_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = LLMClient(
        settings=make_settings(tmp_path, provider="anthropic", anthropic_api_key=None)
    )
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        _ = client.anthropic


def test_missing_openai_key_is_reported_before_any_client_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient(settings=make_settings(tmp_path, provider="openai", openai_api_key=None))
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _ = client.openai


# ------------------------------------------------------- hermeticity guard


def test_the_hermeticity_guard_blocks_non_loopback_connections() -> None:
    """Proves the suite cannot silently start reaching a provider."""
    import socket as socket_module

    from tests.conftest import NetworkAccessDenied

    with pytest.raises(NetworkAccessDenied):
        socket_module.create_connection(("api.anthropic.com", 443), timeout=1)
    with pytest.raises(NetworkAccessDenied):
        socket_module.getaddrinfo("api.anthropic.com", 443)
    with (
        socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM) as probe,
        pytest.raises(NetworkAccessDenied),
    ):
        probe.connect(("api.anthropic.com", 443))


# --------------------------------------------------------------- extract_json


def test_extract_json_parses_bare_json() -> None:
    assert extract_json_object('{"a": 1, "b": [1, 2]}') == {"a": 1, "b": [1, 2]}


def test_extract_json_parses_a_fenced_block() -> None:
    raw = '```json\n{"passed": true, "summary": "ok"}\n```'
    assert extract_json_object(raw) == {"passed": True, "summary": "ok"}


def test_extract_json_parses_an_unlabelled_fence() -> None:
    assert extract_json_object('```\n{"ok": true}\n```') == {"ok": True}


def test_extract_json_parses_json_wrapped_in_prose() -> None:
    raw = 'Sure! Here is the plan you asked for:\n{"tasks": [], "rationale": "x"}\nHope it helps.'
    assert extract_json_object(raw) == {"tasks": [], "rationale": "x"}


def test_extract_json_ignores_braces_inside_strings() -> None:
    raw = '{"summary": "use a } brace and { this", "n": 2}'
    assert extract_json_object(raw) == {"summary": "use a } brace and { this", "n": 2}


def test_extract_json_handles_escaped_quotes() -> None:
    raw = '{"summary": "he said \\"hi\\" and left", "ok": true}'
    assert extract_json_object(raw) == {"summary": 'he said "hi" and left', "ok": True}


def test_extract_json_tolerates_surrounding_whitespace() -> None:
    assert extract_json_object('   \n {"a": 1} \t ') == {"a": 1}


def test_extract_json_returns_the_first_object_of_several() -> None:
    raw = 'note\n{"first": 1}\nmore\n{"second": 2}'
    assert extract_json_object(raw) == {"first": 1}


def test_extract_json_rejects_plain_prose() -> None:
    with pytest.raises(ValueError, match="Could not extract a JSON object"):
        extract_json_object("I am afraid I cannot help with that.")


def test_extract_json_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty response"):
        extract_json_object("   ")


def test_extract_json_rejects_truncated_json() -> None:
    with pytest.raises(ValueError):
        extract_json_object('{"a": 1')


def test_extract_json_rejects_a_top_level_array() -> None:
    with pytest.raises(ValueError, match="not an object"):
        extract_json_object("[1, 2, 3]")


def test_extract_json_error_message_carries_a_snippet() -> None:
    raw = "sorry, " + ("x" * 400) + " nope"
    with pytest.raises(ValueError) as excinfo:
        extract_json_object(raw)
    message = str(excinfo.value)
    assert "sorry" in message
    assert "chars)" in message  # the snippet is truncated, not dumped whole
    assert len(message) < 1200


def test_extract_json_rejects_a_non_string() -> None:
    with pytest.raises(TypeError):
        extract_json_object(b'{"a": 1}')  # type: ignore[arg-type]


def test_client_extract_json_delegates_to_the_module_function(tmp_path: Path) -> None:
    client = LLMClient(settings=make_settings(tmp_path))
    assert client.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        client.extract_json("nope")


# --------------------------------------------------------- is_retryable_error


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 529])
def test_transient_status_codes_are_retryable(status: int) -> None:
    assert is_retryable_error(StatusError(status)) is True


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 405, 413, 422])
def test_client_error_status_codes_are_not_retryable(status: int) -> None:
    assert is_retryable_error(StatusError(status)) is False


def test_sdk_authentication_errors_are_not_retryable() -> None:
    assert is_retryable_error(AuthenticationError("invalid x-api-key")) is False


def test_sdk_rate_limit_errors_are_retryable() -> None:
    assert is_retryable_error(RateLimitError("slow down")) is True


def test_httpx_transport_errors_are_retryable() -> None:
    assert is_retryable_error(httpx.ConnectError("connection refused")) is True
    assert is_retryable_error(httpx.ReadTimeout("slow")) is True
    assert is_retryable_error(httpx.RemoteProtocolError("peer closed")) is True


def test_asyncio_timeout_error_is_retryable() -> None:
    assert is_retryable_error(asyncio.TimeoutError()) is True


def test_value_error_is_not_retryable() -> None:
    assert is_retryable_error(ValueError("could not extract JSON")) is False


def test_status_is_read_from_a_nested_response() -> None:
    exc = httpx.HTTPStatusError(
        "boom",
        request=httpx.Request("POST", "http://ollama.invalid"),
        response=httpx.Response(429),
    )
    assert is_retryable_error(exc) is True


# --------------------------------------------------------------------- retries


async def test_complete_retries_a_transient_failure_then_succeeds(tmp_path: Path) -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json={"message": {"content": "recovered"}})

    client = ollama_client(handler, tmp_path, retry_attempts=4)
    try:
        assert await client.complete("prompt") == "recovered"
    finally:
        await client.aclose()
    assert len(attempts) == 3


async def test_complete_gives_up_after_retry_attempts(tmp_path: Path) -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(500, text="always broken")

    client = ollama_client(handler, tmp_path, retry_attempts=3)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.complete("prompt")
    finally:
        await client.aclose()
    assert len(attempts) == 3


async def test_complete_does_not_retry_an_auth_error(tmp_path: Path) -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(401, text="invalid api key")

    client = ollama_client(handler, tmp_path, retry_attempts=5)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.complete("prompt")
    finally:
        await client.aclose()
    assert len(attempts) == 1, "an auth error must fail fast"


async def test_complete_propagates_a_non_retryable_sdk_error_after_one_attempt(
    tmp_path: Path,
) -> None:
    fake = FakeOpenAI([AuthenticationError("bad key")] * 5)
    client = LLMClient(settings=make_settings(tmp_path, provider="openai", retry_attempts=5))
    client._openai = fake
    with pytest.raises(AuthenticationError):
        await client.complete("prompt")
    assert len(fake.completions.kwargs) == 1


async def test_complete_retries_a_rate_limit_then_succeeds(tmp_path: Path) -> None:
    fake = FakeOpenAI([RateLimitError("429"), FakeOpenAIResponse("done")])
    client = LLMClient(settings=make_settings(tmp_path, provider="openai", retry_attempts=3))
    client._openai = fake
    assert await client.complete("prompt") == "done"
    assert len(fake.completions.kwargs) == 2


async def test_retry_attempts_of_one_never_retries(tmp_path: Path) -> None:
    fake = FakeOpenAI([RateLimitError("429"), FakeOpenAIResponse("done")])
    client = LLMClient(settings=make_settings(tmp_path, provider="openai", retry_attempts=1))
    client._openai = fake
    with pytest.raises(RateLimitError):
        await client.complete("prompt")
    assert len(fake.completions.kwargs) == 1


# ----------------------------------------------------------- anthropic parsing


def test_anthropic_text_joins_every_text_block() -> None:
    response = FakeAnthropicResponse(
        [FakeBlock("text", "first "), FakeBlock("text", "second"), FakeBlock("text", " third")]
    )
    assert LLMClient._anthropic_text(response, "model-x") == "first second third"


def test_anthropic_text_skips_non_text_blocks() -> None:
    """A leading ``thinking`` block must not break ``content[0]`` handling."""
    response = FakeAnthropicResponse(
        [FakeBlock("thinking", "let me think"), FakeBlock("text", "the answer")]
    )
    assert LLMClient._anthropic_text(response, "model-x") == "the answer"


def test_anthropic_text_reads_dict_blocks() -> None:
    response = FakeAnthropicResponse([{"type": "text", "text": "d1"}, {"type": "tool_use"}])
    assert LLMClient._anthropic_text(response, "model-x") == "d1"


def test_anthropic_text_without_any_text_raises_a_helpful_error() -> None:
    response = FakeAnthropicResponse([FakeBlock("thinking", "hmm"), FakeBlock("tool_use")])
    with pytest.raises(ValueError, match="no text content"):
        LLMClient._anthropic_text(response, "model-x")


def test_anthropic_text_with_empty_content_raises() -> None:
    with pytest.raises(ValueError, match="no text content"):
        LLMClient._anthropic_text(FakeAnthropicResponse([]), "model-x")


async def test_anthropic_complete_joins_blocks_and_sends_settings(tmp_path: Path) -> None:
    fake = FakeAnthropic(
        [FakeAnthropicResponse([FakeBlock("thinking", "hmm"), FakeBlock("text", "final answer")])]
    )
    settings = make_settings(
        tmp_path,
        provider="anthropic",
        anthropic_api_key="test-key",
        llm_max_tokens=321,
        llm_temperature=0.25,
    )
    client = LLMClient(settings=settings)
    client._anthropic = fake
    assert await client.complete("prompt", system="be terse") == "final answer"
    kwargs = fake.messages.kwargs[0]
    assert kwargs["max_tokens"] == 321
    assert kwargs["temperature"] == 0.25
    assert kwargs["system"] == "be terse"
    assert kwargs["messages"] == [{"role": "user", "content": "prompt"}]


# ----------------------------------------------------------------- openai path


async def test_openai_complete_sends_token_budget_and_temperature(tmp_path: Path) -> None:
    fake = FakeOpenAI([FakeOpenAIResponse("openai says hi")])
    settings = make_settings(
        tmp_path,
        provider="openai",
        openai_api_key="test-key",
        llm_max_tokens=777,
        llm_temperature=0.9,
        model="gpt-x",
    )
    client = LLMClient(settings=settings)
    client._openai = fake
    assert await client.complete("prompt", system="sys") == "openai says hi"
    kwargs = fake.completions.kwargs[0]
    assert kwargs["model"] == "gpt-x"
    assert kwargs["max_tokens"] == 777
    assert kwargs["temperature"] == 0.9
    assert kwargs["messages"][0] == {"role": "system", "content": "sys"}


async def test_openai_empty_choices_raise(tmp_path: Path) -> None:
    empty = type("Response", (), {"choices": []})()
    fake = FakeOpenAI([empty])
    client = LLMClient(settings=make_settings(tmp_path, provider="openai"))
    client._openai = fake
    with pytest.raises(ValueError, match="empty `choices`"):
        await client.complete("prompt")


# ----------------------------------------------------------------- ollama path


async def test_ollama_complete_sends_options_and_model(tmp_path: Path) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ollama says hi"}})

    client = ollama_client(
        handler,
        tmp_path,
        llm_max_tokens=555,
        llm_temperature=0.75,
        model="llama-x",
        ollama_base_url="http://ollama.invalid:11434",
    )
    try:
        assert await client.complete("prompt") == "ollama says hi"
    finally:
        await client.aclose()
    assert bodies[0]["model"] == "llama-x"
    assert bodies[0]["options"] == {"temperature": 0.75, "num_predict": 555}
    assert bodies[0]["stream"] is False


def test_ollama_chat_url_is_derived_from_the_base_url(tmp_path: Path) -> None:
    client = LLMClient(
        settings=make_settings(tmp_path, provider="ollama", ollama_base_url="http://host:1234/")
    )
    assert client.ollama_chat_url == "http://host:1234/api/chat"


async def test_ollama_non_200_raises(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model not found")

    client = ollama_client(handler, tmp_path, retry_attempts=1)
    try:
        with pytest.raises(httpx.HTTPStatusError, match="HTTP 404"):
            await client.complete("prompt")
    finally:
        await client.aclose()


async def test_ollama_non_json_body_raises(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>nope</html>")

    client = ollama_client(handler, tmp_path)
    try:
        with pytest.raises(ValueError, match="non-JSON"):
            await client.complete("prompt")
    finally:
        await client.aclose()


async def test_ollama_missing_content_raises(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "   "}})

    client = ollama_client(handler, tmp_path)
    try:
        with pytest.raises(ValueError, match="no message content"):
            await client.complete("prompt")
    finally:
        await client.aclose()


async def test_an_unsupported_provider_raises(tmp_path: Path) -> None:
    from furrow.config import Settings

    bogus = Settings(provider="anthropic")
    object.__setattr__(bogus, "provider", "vertex")
    client = LLMClient(settings=make_settings(tmp_path))
    client.settings = bogus
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("prompt")


# ---------------------------------------------------------------------- files


def test_list_files_returns_forward_slash_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "pkg" / "deep").mkdir(parents=True)
    (tmp_path / "pkg" / "deep" / "mod.py").write_text("x = 1")
    (tmp_path / "top.py").write_text("y = 2")
    client = LLMClient(settings=make_settings(tmp_path))
    listed = client.list_files(tmp_path)
    assert sorted(listed) == ["pkg/deep/mod.py", "top.py"]
    assert all("\\" not in path for path in listed)
    assert not any(path.startswith("/") for path in listed)


def test_list_files_skips_noise_directories(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("k")
    for name in ("__pycache__", "node_modules", ".git", "dist", "build", ".venv"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "junk.py").write_text("junk")
    egg = tmp_path / "thing.egg-info"
    egg.mkdir()
    (egg / "meta.py").write_text("junk")
    client = LLMClient(settings=make_settings(tmp_path))
    assert client.list_files(tmp_path) == ["keep.py"]


def test_list_files_skips_bytecode(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("x")
    (tmp_path / "mod.pyc").write_text("junk")
    (tmp_path / "mod.pyo").write_text("junk")
    client = LLMClient(settings=make_settings(tmp_path))
    assert client.list_files(tmp_path) == ["mod.py"]


def test_list_files_returns_empty_for_a_missing_directory(tmp_path: Path) -> None:
    client = LLMClient(settings=make_settings(tmp_path))
    assert client.list_files(tmp_path / "does-not-exist") == []


def test_skip_directories_contains_the_usual_noise() -> None:
    for name in ("__pycache__", "node_modules", ".git", ".venv", "dist"):
        assert name in SKIP_DIRECTORIES


async def test_write_file_creates_parent_directories(tmp_path: Path) -> None:
    client = LLMClient(settings=make_settings(tmp_path))
    target = tmp_path / "deep" / "nested" / "mod.py"
    await client.write_file(target, "content = 1\n")
    assert target.read_text() == "content = 1\n"
    assert await client.read_file(target) == "content = 1\n"


async def test_write_file_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "mod.py"
    target.write_text("old")
    client = LLMClient(settings=make_settings(tmp_path))
    await client.write_file(target, "new")
    assert target.read_text() == "new"


async def test_read_file_raises_for_a_missing_file(tmp_path: Path) -> None:
    client = LLMClient(settings=make_settings(tmp_path))
    with pytest.raises(OSError):
        await client.read_file(tmp_path / "missing.py")


# --------------------------------------------------------------------- aclose


async def test_aclose_is_idempotent_and_drops_the_client(tmp_path: Path) -> None:
    client = ollama_client(
        lambda request: httpx.Response(200, json={"message": {"content": "x"}}),
        tmp_path,
    )
    assert client._http is not None
    await client.aclose()
    assert client._http is None
    await client.aclose()  # second call is a no-op
    assert client._http is None


async def test_aclose_without_a_client_is_a_no_op(tmp_path: Path) -> None:
    client = LLMClient(settings=make_settings(tmp_path))
    await client.aclose()
    assert client._http is None


async def test_aclose_survives_a_cancellation(tmp_path: Path) -> None:
    """The close completes even though the caller is unwinding from a cancel."""
    client = ollama_client(
        lambda request: httpx.Response(200, json={"message": {"content": "x"}}),
        tmp_path,
    )
    http = client._http
    assert http is not None
    started = asyncio.Event()
    release = asyncio.Event()
    real_aclose = http.aclose

    async def slow_aclose() -> None:
        started.set()
        await release.wait()
        await real_aclose()

    http.aclose = slow_aclose  # type: ignore[method-assign]

    closer = asyncio.ensure_future(client.aclose())
    await asyncio.wait_for(started.wait(), timeout=5)
    assert client._http is None, "the reference is dropped before awaiting"
    closer.cancel()
    release.set()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(closer, timeout=5)
    assert http.is_closed, "the sockets must really be released"
