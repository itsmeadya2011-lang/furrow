"""Regression tests for :mod:`furrow.llm`.

Every test here is hermetic: no network, no credentials, no sleeping. The provider SDK clients
are replaced by fakes that record the kwargs they were called with and return canned
responses, and the retry tests replace the backoff with tenacity's no-op wait so they assert
attempt *counts* in microseconds. ``tests/conftest.py`` already clears every ``FURROW_`` and
vendor environment variable, so a developer's real API key cannot change an outcome here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import anthropic
import httpx
import openai
import pytest
from tenacity import RetryError, stop_after_attempt, wait_none, wait_random_exponential

from furrow import llm
from furrow.config import Provider, Settings, get_settings
from furrow.llm import (
    LLMClient,
    LLMError,
    LLMResponseTruncated,
    ProviderAuthError,
    ProviderUnavailableError,
)

#: Distinctive values used as credentials so a test can assert, by substring search, that a
#: secret never reaches an exception message.
ANTHROPIC_SECRET = "anthropic-secret-do-not-log"
OPENAI_SECRET = "openai-secret-do-not-log"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeBlock:
    """Stand-in for one Anthropic content block.

    ``text`` is only set when the block type has one, which is exactly the trap the old
    ``response.content[0].text`` fell into: a ``ThinkingBlock`` or ``ToolUseBlock`` has no
    ``.text`` attribute at all.
    """

    def __init__(self, type_: str, text: str | None = None) -> None:
        self.type = type_
        if text is not None:
            self.text = text


class _FakeAnthropicMessage:
    """The subset of ``anthropic.types.Message`` that the client reads."""

    def __init__(self, content: list[_FakeBlock], stop_reason: str = "end_turn") -> None:
        self.content = content
        self.stop_reason = stop_reason


class _FakeAnthropicMessages:
    def __init__(
        self,
        result: _FakeAnthropicMessage | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result

    async def close(self) -> None:
        self.closed = True


class _FakeAnthropic:
    def __init__(self, messages: _FakeAnthropicMessages) -> None:
        self.messages = messages
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


class _FakeOpenAIMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None, finish_reason: str = "stop") -> None:
        self.message = _FakeOpenAIMessage(content)
        self.finish_reason = finish_reason


class _FakeCompletions:
    def __init__(
        self,
        choices: list[_FakeChoice] | None = None,
        error: Exception | None = None,
        failures_before_success: int = 0,
    ) -> None:
        self.choices = choices if choices is not None else [_FakeChoice("ok")]
        self.error = error
        self.failures_before_success = failures_before_success
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.failures_before_success > 0:
            self.failures_before_success -= 1
            raise _status_error(openai.InternalServerError, 503)
        if self.error is not None:
            raise self.error
        return type("_R", (), {"choices": self.choices})()

    async def close(self) -> None:
        self.closed = True


class _FakeOpenAI:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = type("_Chat", (), {"completions": completions})()
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(**overrides: Any) -> Settings:
    """Settings that cannot see the ambient environment or a stray ``.env``."""
    return Settings(_env_file=None, **overrides)


def _anthropic_settings(**overrides: Any) -> Settings:
    return _settings(
        provider=Provider.ANTHROPIC, anthropic_api_key=ANTHROPIC_SECRET, **overrides
    )


def _openai_settings(**overrides: Any) -> Settings:
    return _settings(provider=Provider.OPENAI, openai_api_key=OPENAI_SECRET, **overrides)


def _anthropic_client(
    settings: Settings,
    result: _FakeAnthropicMessage | None = None,
    error: Exception | None = None,
) -> tuple[LLMClient, _FakeAnthropicMessages]:
    """An ``LLMClient`` whose Anthropic branch is a recording fake."""
    messages = _FakeAnthropicMessages(result=result, error=error)
    client = LLMClient(settings)
    client._anthropic = _FakeAnthropic(messages)
    return client, messages


def _openai_client(
    settings: Settings,
    choices: list[_FakeChoice] | None = None,
    error: Exception | None = None,
) -> tuple[LLMClient, _FakeCompletions]:
    """An ``LLMClient`` whose OpenAI branch is a recording fake."""
    completions = _FakeCompletions(choices=choices, error=error)
    client = LLMClient(settings)
    client._openai = _FakeOpenAI(completions)
    return client, completions


def _status_error(cls: type[BaseException], status: int) -> BaseException:
    """A real SDK status error carrying a synthetic response, so ``status_code`` is genuine."""
    return cls(
        f"provider answered {status}",
        response=httpx.Response(status, request=httpx.Request("POST", "https://example.test")),
        body=None,
    )


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make retries instant by swapping the module's wait strategy for a no-op.

    The retry policy is built per call from a module-level name, so patching
    ``furrow.llm.wait_random_exponential`` keeps the real stop/retry/reraise configuration
    under test while removing the seconds of sleeping.
    """
    monkeypatch.setattr(llm, "wait_random_exponential", lambda **_kwargs: wait_none())


# ---------------------------------------------------------------------------
# Construction: the import-time default argument bug
# ---------------------------------------------------------------------------


def test_none_settings_resolves_to_process_settings() -> None:
    """``LLMClient()`` and ``LLMClient(None)`` store real Settings, never ``None``.

    Regression test for the load-bearing bug: the old signature was
    ``def __init__(self, settings: Settings = settings)``, so every agent that passed
    ``settings=None`` stored ``None`` and the first ``self.settings.model`` raised
    ``AttributeError`` in the middle of a run.
    """
    default_client = LLMClient()
    none_client = LLMClient(None)

    assert none_client.settings is not None
    assert isinstance(none_client.settings, Settings)
    assert none_client.settings is get_settings()
    assert default_client.settings is none_client.settings
    # The real regression check: settings are usable, not just present.
    assert none_client.settings.model_for("worker") == get_settings().model_for("worker")


def test_explicit_settings_is_used_verbatim() -> None:
    """A passed-in Settings is stored as-is, never replaced by the global singleton."""
    custom = _openai_settings(max_tokens=42, model="gpt-test")
    client = LLMClient(custom)

    assert client.settings is custom
    assert client.settings is not get_settings()
    assert client.settings.max_tokens == 42
    assert client.settings.model_for("planner") == "gpt-test"


def test_construction_touches_no_provider_client() -> None:
    """No SDK client exists until one is asked for, so no key is needed to construct."""
    client = LLMClient(_settings())

    assert client._anthropic is None
    assert client._openai is None
    assert client._ollama is None


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


async def test_missing_anthropic_key_raises_provider_auth_error() -> None:
    client = LLMClient(_settings(provider=Provider.ANTHROPIC))
    with pytest.raises(ProviderAuthError) as exc_info:
        await client.complete("hi")
    assert "ANTHROPIC_API_KEY" in str(exc_info.value)


async def test_missing_openai_key_raises_provider_auth_error() -> None:
    client = LLMClient(_settings(provider=Provider.OPENAI))
    with pytest.raises(ProviderAuthError) as exc_info:
        await client.complete("hi")
    assert "OPENAI_API_KEY" in str(exc_info.value)


def test_key_falls_back_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key exported after Settings was built is still found.

    Settings snapshots the environment at construction, so a client holding older settings
    only works because resolution consults ``os.environ`` as well.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-environment")
    settings = _settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings)

    assert client._api_key(Provider.ANTHROPIC) == "from-the-environment"
    assert client.anthropic.api_key == "from-the-environment"


def test_configured_key_wins_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-the-environment")
    settings = _settings(provider=Provider.OPENAI, openai_api_key=OPENAI_SECRET)

    assert LLMClient(settings)._api_key(Provider.OPENAI) == OPENAI_SECRET


def test_ollama_key_is_a_placeholder() -> None:
    """Ollama ignores the Authorization header, but the OpenAI SDK demands a non-empty key."""
    client = LLMClient(_settings(provider=Provider.OLLAMA))

    assert client._api_key(Provider.OLLAMA) == "ollama"
    assert client.ollama.api_key == "ollama"


async def test_rejected_secret_is_reported_by_status_never_by_value() -> None:
    """HTTP 401 is a credential problem; the message names the status, never the key."""
    client, _ = _anthropic_client(
        _anthropic_settings(), error=_status_error(anthropic.AuthenticationError, 401)
    )

    with pytest.raises(ProviderAuthError) as exc_info:
        await client.complete("hi")

    message = str(exc_info.value)
    assert "401" in message
    assert ANTHROPIC_SECRET not in message
    assert "claude" in message


async def test_secret_is_scrubbed_from_arbitrary_failure_text() -> None:
    """Even a provider error that quotes the key has it redacted before it is re-raised."""
    client, _ = _anthropic_client(
        _anthropic_settings(), error=RuntimeError(f"upstream echoed {ANTHROPIC_SECRET}")
    )
    with pytest.raises(LLMError) as exc_info:
        await client.complete("hi")

    assert ANTHROPIC_SECRET not in str(exc_info.value)
    assert "***" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


def test_retry_policy_is_built_from_settings() -> None:
    """The attempt budget, backoff bounds and reraise flag come from settings, per call."""
    client = LLMClient(_settings(llm_max_retries=5))
    policy = client._retrying()

    assert isinstance(policy.stop, stop_after_attempt)
    # +1 because the setting counts retries; 0 must still mean exactly one attempt.
    assert policy.stop.max_attempt_number == 6
    assert isinstance(policy.wait, wait_random_exponential)
    assert policy.reraise is True
    # A different instance gets a different budget: the policy is not frozen at import time.
    assert LLMClient(_settings(llm_max_retries=0))._retrying().stop.max_attempt_number == 1


async def test_transient_failure_is_retried_exactly_llm_max_retries_times(
    no_backoff: None,
) -> None:
    """A transient failure consumes the whole budget and then surfaces, typed."""
    settings = _openai_settings(llm_max_retries=2)
    client, completions = _openai_client(
        settings, error=_status_error(openai.RateLimitError, 429)
    )

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await client.complete("hi")

    assert len(completions.calls) == settings.llm_max_retries + 1
    # reraise=True: the caller sees the typed error, not tenacity.RetryError.
    assert not isinstance(exc_info.value, RetryError)


async def test_auth_error_is_attempted_exactly_once(no_backoff: None) -> None:
    """Retrying a bad credential would waste the entire cycle to reach the same failure."""
    settings = _anthropic_settings(llm_max_retries=3)
    client, messages = _anthropic_client(
        settings, error=_status_error(anthropic.AuthenticationError, 401)
    )

    with pytest.raises(ProviderAuthError):
        await client.complete("hi")

    assert len(messages.calls) == 1


async def test_truncated_response_is_not_retried(no_backoff: None) -> None:
    """The request succeeded; replaying it unchanged truncates identically."""
    settings = _anthropic_settings(llm_max_retries=3)
    client, messages = _anthropic_client(
        settings,
        result=_FakeAnthropicMessage([_FakeBlock("thinking")], stop_reason="max_tokens"),
    )

    with pytest.raises(LLMResponseTruncated):
        await client.complete("hi")

    assert len(messages.calls) == 1


async def test_zero_retries_means_exactly_one_attempt(no_backoff: None) -> None:
    settings = _openai_settings(llm_max_retries=0)
    client, completions = _openai_client(
        settings, error=_status_error(openai.RateLimitError, 429)
    )

    with pytest.raises(ProviderUnavailableError):
        await client.complete("hi")

    assert len(completions.calls) == 1


async def test_transient_failure_recovers_on_a_later_attempt(no_backoff: None) -> None:
    """The point of retrying: a blip does not have to end the cycle."""
    settings = _openai_settings(llm_max_retries=3)
    completions = _FakeCompletions(failures_before_success=2)
    client = LLMClient(settings)
    client._openai = _FakeOpenAI(completions)

    assert await client.complete("hi") == "ok"
    assert len(completions.calls) == 3


@pytest.mark.parametrize(
    ("error_class", "status", "expected"),
    [
        (anthropic.AuthenticationError, 401, ProviderAuthError),
        (anthropic.PermissionDeniedError, 403, ProviderAuthError),
        (anthropic.RateLimitError, 429, ProviderUnavailableError),
        (anthropic.InternalServerError, 503, ProviderUnavailableError),
        (anthropic.BadRequestError, 400, LLMError),
        (anthropic.NotFoundError, 404, LLMError),
    ],
)
async def test_anthropic_status_codes_map_to_the_typed_hierarchy(
    no_backoff: None, error_class: type[BaseException], status: int, expected: type[LLMError]
) -> None:
    """Only 401/403 and the transient set get the retryable/permanent treatment."""
    settings = _anthropic_settings(llm_max_retries=3)
    client, messages = _anthropic_client(
        settings, error=_status_error(error_class, status)
    )

    with pytest.raises(expected) as exc_info:
        await client.complete("hi")

    assert str(status) in str(exc_info.value)
    # Only the transient class may be replayed; 4xx and auth failures are one-shot.
    expected_calls = settings.llm_max_retries + 1 if expected is ProviderUnavailableError else 1
    assert len(messages.calls) == expected_calls


@pytest.mark.parametrize(
    "error",
    [
        anthropic.APIConnectionError(request=httpx.Request("POST", "https://example.test")),
        anthropic.APITimeoutError(request=httpx.Request("POST", "https://example.test")),
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("too slow"),
        TimeoutError("no answer"),
    ],
)
async def test_reachability_failures_are_transient(
    no_backoff: None, error: Exception
) -> None:
    """Anything that never reached the provider is transient by definition."""
    settings = _anthropic_settings(llm_max_retries=1)
    client, messages = _anthropic_client(settings, error=error)

    with pytest.raises(ProviderUnavailableError):
        await client.complete("hi")

    assert len(messages.calls) == 2


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


async def test_unknown_provider_raises_llm_error_naming_it() -> None:
    client = LLMClient(_anthropic_settings())
    # Provider is a str enum, so the only way to reach the fallthrough is to bypass
    # pydantic's validation, exactly as a corrupted settings object would.
    client.settings.__dict__["provider"] = "not-a-provider"

    with pytest.raises(LLMError) as exc_info:
        await client.complete("hi")

    assert "not-a-provider" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Anthropic branch
# ---------------------------------------------------------------------------


async def test_anthropic_max_tokens_defaults_to_settings() -> None:
    settings = _anthropic_settings(max_tokens=1234)
    client, messages = _anthropic_client(
        settings, result=_FakeAnthropicMessage([_FakeBlock("text", "ok")])
    )

    await client.complete("hi")
    assert messages.calls[0]["max_tokens"] == 1234

    await client.complete("hi", max_tokens=16)
    assert messages.calls[1]["max_tokens"] == 16


async def test_anthropic_model_resolution() -> None:
    settings = _anthropic_settings(model="claude-test")
    client, messages = _anthropic_client(
        settings, result=_FakeAnthropicMessage([_FakeBlock("text", "ok")])
    )

    await client.complete("hi")
    assert messages.calls[0]["model"] == "claude-test"

    await client.complete("hi", model="claude-explicit")
    assert messages.calls[1]["model"] == "claude-explicit"


async def test_anthropic_omits_system_when_falsy() -> None:
    """An empty `system` is omitted entirely: the SDK rejects "" and no caller wants a filler."""
    client, messages = _anthropic_client(
        _anthropic_settings(), result=_FakeAnthropicMessage([_FakeBlock("text", "ok")])
    )

    await client.complete("hi", system=None)
    assert "system" not in messages.calls[0]

    await client.complete("hi", system="")
    assert "system" not in messages.calls[1]

    await client.complete("hi", system="you are a pirate")
    assert messages.calls[2]["system"] == "you are a pirate"


async def test_anthropic_concatenates_text_across_blocks_and_skips_non_text() -> None:
    """A leading thinking block has no ``.text``; the old ``content[0].text`` raised."""
    client, messages = _anthropic_client(
        _anthropic_settings(),
        result=_FakeAnthropicMessage(
            [
                _FakeBlock("thinking"),
                _FakeBlock("tool_use"),
                _FakeBlock("text", "hello "),
                _FakeBlock("text", "world"),
            ]
        ),
    )

    assert await client.complete("hi") == "hello world"
    assert messages.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


async def test_anthropic_max_tokens_stop_with_no_text_is_truncated() -> None:
    client, _ = _anthropic_client(
        _anthropic_settings(),
        result=_FakeAnthropicMessage([_FakeBlock("thinking")], stop_reason="max_tokens"),
    )

    with pytest.raises(LLMResponseTruncated) as exc_info:
        await client.complete("hi")

    assert "max_tokens" in str(exc_info.value)


async def test_anthropic_max_tokens_stop_keeps_the_partial_text() -> None:
    """Truncation is only an error when there is nothing usable at all."""
    client, _ = _anthropic_client(
        _anthropic_settings(),
        result=_FakeAnthropicMessage([_FakeBlock("text", "partial")], stop_reason="max_tokens"),
    )

    assert await client.complete("hi") == "partial"


async def test_anthropic_empty_text_with_normal_stop_returns_empty_string() -> None:
    """The contract is total; the caller's JSON parse fails loudly on the real reason."""
    client, _ = _anthropic_client(
        _anthropic_settings(), result=_FakeAnthropicMessage([_FakeBlock("thinking")])
    )

    assert await client.complete("hi") == ""


# ---------------------------------------------------------------------------
# OpenAI branch
# ---------------------------------------------------------------------------


async def test_openai_sends_no_system_message_when_falsy() -> None:
    client, completions = _openai_client(_openai_settings())

    await client.complete("hi", system=None)
    assert completions.calls[0]["messages"] == [{"role": "user", "content": "hi"}]

    await client.complete("hi", system="")
    assert completions.calls[1]["messages"] == [{"role": "user", "content": "hi"}]


async def test_openai_sends_system_message_when_truthy() -> None:
    client, completions = _openai_client(_openai_settings())

    await client.complete("hi", system="be terse")

    assert completions.calls[0]["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


async def test_openai_returns_empty_string_for_null_content() -> None:
    """A refusal or tool-only turn yields ``None`` content; that becomes ""."""
    client, _ = _openai_client(_openai_settings(), choices=[_FakeChoice(None)])

    assert await client.complete("hi") == ""


async def test_openai_returns_empty_string_when_there_are_no_choices() -> None:
    client, _ = _openai_client(_openai_settings(), choices=[])

    assert await client.complete("hi") == ""


async def test_openai_max_tokens_and_model_resolution() -> None:
    settings = _openai_settings(model="gpt-test", max_tokens=77)
    client, completions = _openai_client(settings)

    await client.complete("hi")
    assert completions.calls[0]["max_tokens"] == 77
    assert completions.calls[0]["model"] == "gpt-test"

    await client.complete("hi", model="gpt-explicit", max_tokens=5)
    assert completions.calls[1]["model"] == "gpt-explicit"
    assert completions.calls[1]["max_tokens"] == 5


# ---------------------------------------------------------------------------
# Ollama branch
# ---------------------------------------------------------------------------


async def test_ollama_uses_an_openai_compatible_client_with_configured_base_url() -> None:
    settings = _settings(
        provider=Provider.OLLAMA,
        ollama_base_url="http://ollama.test:11434",
        ollama_model="llama3.1",
        request_timeout=11.0,
    )
    client = LLMClient(settings)

    # Real SDK construction only: no request is made, and no credential is required.
    assert str(client.ollama.base_url).rstrip("/") == "http://ollama.test:11434"
    assert client.ollama.api_key == "ollama"
    assert client.ollama.timeout == 11.0


async def test_ollama_defaults_the_model_and_omits_system_when_falsy() -> None:
    settings = _settings(
        provider=Provider.OLLAMA,
        ollama_base_url="http://ollama.test:11434",
        ollama_model="llama3.1",
    )
    completions = _FakeCompletions(choices=[_FakeChoice("hi!")])
    client = LLMClient(settings)
    client._ollama = _FakeOpenAI(completions)

    assert await client.complete("hello") == "hi!"
    assert completions.calls[0]["model"] == "llama3.1"
    assert completions.calls[0]["messages"] == [{"role": "user", "content": "hello"}]

    await client.complete("hello", system="be terse", model="llama-explicit")
    assert completions.calls[1]["model"] == "llama-explicit"
    assert completions.calls[1]["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"},
    ]


async def test_ollama_unreachable_daemon_is_retried_then_raised(no_backoff: None) -> None:
    settings = _settings(
        provider=Provider.OLLAMA,
        llm_max_retries=2,
        ollama_base_url="http://ollama.test:11434",
    )
    completions = _FakeCompletions(
        error=httpx.ConnectError("connection refused")
    )
    client = LLMClient(settings)
    client._ollama = _FakeOpenAI(completions)

    with pytest.raises(ProviderUnavailableError):
        await client.complete("hello")

    assert len(completions.calls) == 3


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def test_list_files_returns_sorted_relative_paths(tmp_path: Path) -> None:
    for relative in ("z.py", "a.py", "pkg/deep/m.py", "notes.md"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# x", encoding="utf-8")

    client = LLMClient(_settings())
    found = client.list_files(tmp_path)

    assert found == ["a.py", "notes.md", "pkg/deep/m.py", "z.py"]
    assert all(not Path(entry).is_absolute() for entry in found)


def test_list_files_honours_the_pattern(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "top.py").write_text("# y", encoding="utf-8")
    (tmp_path / "pkg" / "deep.py").write_text("# y", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# z", encoding="utf-8")

    client = LLMClient(_settings())

    # fnmatch's "*" spans "/", so "*.py" reaches every depth, as rglob("*") used to.
    assert client.list_files(tmp_path, pattern="*.py") == ["pkg/deep.py", "top.py"]
    assert client.list_files(tmp_path, "notes.*") == ["notes.md"]


def test_list_files_skips_junk_directories(tmp_path: Path) -> None:
    junk = (
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
    )
    for name in junk:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "leaked.py").write_text("# nope", encoding="utf-8")
    (tmp_path / "real.py").write_text("# yes", encoding="utf-8")

    assert LLMClient(_settings()).list_files(tmp_path) == ["real.py"]


def test_list_files_returns_empty_for_missing_directory(tmp_path: Path) -> None:
    """A path that does not exist is a normal answer for an agent, not an error."""
    assert LLMClient(_settings()).list_files(tmp_path / "nope") == []
    assert LLMClient(_settings()).list_files(tmp_path / "nope", "*.py") == []


def test_list_files_does_not_raise_on_a_broken_symlink(tmp_path: Path) -> None:
    (tmp_path / "real.py").write_text("# yes", encoding="utf-8")
    (tmp_path / "dangling.py").symlink_to(tmp_path / "missing" / "target.py")

    found = LLMClient(_settings()).list_files(tmp_path)

    assert "real.py" in found


@pytest.mark.skipif(
    getattr(os, "geteuid", lambda: 1)() == 0,
    reason="root bypasses directory permissions, so chmod 000 cannot make the walk fail",
)
def test_list_files_survives_an_unreadable_subdirectory(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "hidden.py").write_text("# nope", encoding="utf-8")
    (tmp_path / "real.py").write_text("# yes", encoding="utf-8")
    locked.chmod(0o000)
    try:
        assert LLMClient(_settings()).list_files(tmp_path) == ["real.py"]
    finally:
        locked.chmod(0o755)


def test_list_files_survives_a_failing_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A subdirectory that cannot be scanned is skipped, not raised.

    ``chmod 000`` cannot demonstrate this when the tests run as root, so the failure is
    injected instead: ``os.walk`` looks ``scandir`` up in the ``os`` module namespace, and its
    documented default is to ignore scan errors rather than abort the walk.
    """
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "hidden.py").write_text("# nope", encoding="utf-8")
    (tmp_path / "real.py").write_text("# yes", encoding="utf-8")
    real_scandir = os.scandir

    def exploding_scandir(path: Any = "*", *args: Any, **kwargs: Any) -> Any:
        if "locked" in os.fspath(path):
            raise PermissionError(13, "Permission denied")
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", exploding_scandir)

    assert LLMClient(_settings()).list_files(tmp_path) == ["real.py"]


async def test_write_file_creates_missing_parents(tmp_path: Path) -> None:
    client = LLMClient(_settings(workspace=tmp_path))
    target = tmp_path / "a" / "b" / "c.txt"

    await client.write_file(target, "h\u00e9llo w\u00f6rld\n")

    assert target.read_text(encoding="utf-8") == "h\u00e9llo w\u00f6rld\n"


async def test_write_file_resolves_relative_paths_under_the_workspace(tmp_path: Path) -> None:
    """A relative path lands in the project even if the process CWD is somewhere else."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = LLMClient(_settings(workspace=workspace))

    await client.write_file("nested/out.txt", "data")

    assert (workspace / "nested" / "out.txt").read_text(encoding="utf-8") == "data"


async def test_read_file_resolves_relative_paths_under_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "in.txt").write_text("contents", encoding="utf-8")
    client = LLMClient(_settings(workspace=workspace))

    assert await client.read_file("in.txt") == "contents"
    assert await client.read_file(workspace / "in.txt") == "contents"


async def test_read_file_missing_raises_oserror(tmp_path: Path) -> None:
    client = LLMClient(_settings(workspace=tmp_path))

    with pytest.raises(OSError):
        await client.read_file("missing.txt")


def test_list_files_resolves_a_relative_directory_under_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "pkg").mkdir(parents=True)
    (workspace / "pkg" / "mod.py").write_text("# x", encoding="utf-8")
    client = LLMClient(_settings(workspace=workspace))

    assert client.list_files(".") == ["pkg/mod.py"]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_aclose_is_safe_and_idempotent_with_no_clients() -> None:
    client = LLMClient(_settings())
    await client.aclose()
    await client.aclose()


async def test_aclose_closes_each_built_client_once() -> None:
    client = LLMClient(_anthropic_settings())
    anthropic_fake = _FakeAnthropic(_FakeAnthropicMessages())
    openai_fake = _FakeOpenAI(_FakeCompletions())
    ollama_fake = _FakeOpenAI(_FakeCompletions())
    client._anthropic = anthropic_fake
    client._openai = openai_fake
    client._ollama = ollama_fake

    await client.aclose()
    await client.aclose()

    assert anthropic_fake.close_calls == 1
    assert openai_fake.close_calls == 1
    assert ollama_fake.close_calls == 1
    assert client._anthropic is None
    assert client._openai is None
    assert client._ollama is None


async def test_aclose_survives_a_failing_pool() -> None:
    """One broken pool must not leak the others."""

    class _Broken(_FakeOpenAI):
        async def close(self) -> None:
            raise RuntimeError("pool already closed")

    client = LLMClient(_anthropic_settings())
    good = _FakeAnthropic(_FakeAnthropicMessages())
    client._anthropic = good
    client._openai = _Broken(_FakeCompletions())

    await client.aclose()

    assert good.close_calls == 1


async def test_async_context_manager_closes_clients() -> None:
    async with LLMClient(_anthropic_settings()) as client:
        assert client.anthropic is client.anthropic

    assert client._anthropic is None


async def test_real_clients_carry_the_configured_timeout() -> None:
    """Both SDK clients get ``request_timeout``; an unset timeout hangs until the server acts."""
    settings = _settings(
        provider=Provider.ANTHROPIC,
        anthropic_api_key=ANTHROPIC_SECRET,
        openai_api_key=OPENAI_SECRET,
        request_timeout=42.0,
    )
    client = LLMClient(settings)

    assert client.anthropic.timeout == 42.0
    assert client.openai.timeout == 42.0
    assert client.anthropic.api_key == ANTHROPIC_SECRET
    assert client.openai.api_key == OPENAI_SECRET

    await client.aclose()
