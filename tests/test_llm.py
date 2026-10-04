"""Tests for :mod:`furrow.llm`.

Every test here stubs the provider SDKs (``AsyncAnthropic``/``AsyncOpenAI``) or the settings, so
nothing reaches the network and no real API key is ever required. The retry tests patch
``asyncio.sleep`` so the exponential backoff is asserted but never paid for.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx
import openai
import pytest

import furrow.llm
from furrow.config import Provider, Settings, get_settings, reset_settings
from furrow.llm import DEFAULT_SYSTEM_PROMPT, LLMClient, LLMError
from tests.conftest import build_settings

# --------------------------------------------------------------------------------------
# SDK doubles
# --------------------------------------------------------------------------------------


class _Namespace:
    """Attribute bag standing in for a provider response object."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class FakeBlock:
    """One entry of an Anthropic ``content`` list."""

    def __init__(self, type: str, text: str | None = None) -> None:
        self.type = type
        if text is not None:
            self.text = text


def _request() -> Any:
    # Typed Any on purpose: the SDKs vendor their own httpx build (httpx2), so a real
    # httpx.Response is runtime-correct but never type-compatible with what they declare.
    return httpx.Request("POST", "http://provider.invalid/v1/messages")


def _response(status: int) -> Any:
    return httpx.Response(status, request=_request())


def rate_limited() -> anthropic.RateLimitError:
    return anthropic.RateLimitError("rate limited", response=_response(429), body=None)


def internal_error() -> anthropic.InternalServerError:
    return anthropic.InternalServerError("server exploded", response=_response(500), body=None)


def connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=_request())


def timed_out() -> anthropic.APITimeoutError:
    return anthropic.APITimeoutError(request=_request())


def untyped_status_error(status: int) -> Exception:
    """A plain exception that advertises a retryable HTTP status."""
    error = RuntimeError(f"upstream said {status}")
    error.status_code = status  # type: ignore[attr-defined]
    return error


class FakeSDK:
    """One fake provider SDK client: records construction kwargs and requests."""

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.requests: list[dict[str, Any]] = []
        self.outcomes: list[Any] = []

    def next(self) -> Any:
        if not self.outcomes:
            raise AssertionError(f"{type(self).__name__} has no scripted outcome")
        if len(self.outcomes) == 1:
            return self.outcomes[0]
        return self.outcomes.pop(0)


class AnthropicMessages:
    def __init__(self, owner: FakeAnthropic) -> None:
        self._owner = owner

    async def create(self, **kwargs: Any) -> Any:
        self._owner.requests.append(kwargs)
        outcome = self._owner.next()
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, list):
            return _Namespace(content=outcome)
        return _Namespace(content=[FakeBlock("text", outcome)])


class OpenAICompletions:
    def __init__(self, owner: FakeOpenAI) -> None:
        self._owner = owner

    async def create(self, **kwargs: Any) -> Any:
        self._owner.requests.append(kwargs)
        outcome = self._owner.next()
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, _Namespace):
            return outcome
        message = _Namespace(content=None if outcome is None else outcome)
        return _Namespace(choices=[_Namespace(message=message)])


class FakeAnthropic(FakeSDK):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.messages = AnthropicMessages(self)


class FakeOpenAI(FakeSDK):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.chat = _Namespace(completions=OpenAICompletions(self))


class SDKStub:
    """Handle for a patched provider SDK symbol.

    ``script(...)`` installs the outcomes every request will replay (the last one repeats), and
    ``instances``/``requests`` expose what the client did. Scripting after the call also works
    because the outcome list is shared by reference with every instance.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, attribute: str, factory: type[FakeSDK]):
        self.attribute = attribute
        self.instances: list[FakeSDK] = []
        self.outcomes: list[Any] = []

        def build(**kwargs: Any) -> FakeSDK:
            instance = factory(**kwargs)
            instance.outcomes = self.outcomes
            self.instances.append(instance)
            return instance

        monkeypatch.setattr(furrow.llm, attribute, build)

    def script(self, *outcomes: Any) -> SDKStub:
        """Replace the scripted outcomes; the last one repeats once the queue is drained."""
        self.outcomes[:] = list(outcomes)
        return self

    @property
    def requests(self) -> list[dict[str, Any]]:
        return [request for instance in self.instances for request in instance.requests]

    @property
    def last_request(self) -> dict[str, Any]:
        assert self.requests, f"{self.attribute} never received a request"
        return self.requests[-1]

    @property
    def only_instance(self) -> FakeSDK:
        assert len(self.instances) == 1, f"expected exactly one {self.attribute} client"
        return self.instances[0]


@pytest.fixture
def stub_anthropic(monkeypatch: pytest.MonkeyPatch) -> SDKStub:
    """Patch ``furrow.llm.AsyncAnthropic`` so no Anthropic client is ever constructed."""
    return SDKStub(monkeypatch, "AsyncAnthropic", FakeAnthropic)


@pytest.fixture
def stub_openai(monkeypatch: pytest.MonkeyPatch) -> SDKStub:
    """Patch ``furrow.llm.AsyncOpenAI`` so no OpenAI client is ever constructed."""
    return SDKStub(monkeypatch, "AsyncOpenAI", FakeOpenAI)


@pytest.fixture
def no_sleep(recorded_sleeps: list[float]) -> list[float]:
    """Retry tests must never pay the backoff; this records what tenacity asked for."""
    return recorded_sleeps


# --------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------


def test_client_with_no_arguments_resolves_real_settings() -> None:
    # Regression: LLMClient(settings=None) used to keep None and blow up on first use.
    client = LLMClient()
    assert client.settings is not None
    assert isinstance(client.settings, Settings)
    assert client.settings is get_settings()


def test_explicit_none_settings_is_treated_as_unset() -> None:
    assert LLMClient(settings=None).settings is get_settings()


def test_explicit_settings_are_used_verbatim(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(model="pinned")
    assert LLMClient(settings).settings is settings


def test_settings_are_read_lazily_so_an_env_change_is_picked_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_settings()
    monkeypatch.setenv("FURROW_MODEL", "env-model")
    assert LLMClient().settings.model == "env-model"


def test_each_client_builds_its_own_sdk_instance(
    stub_anthropic: SDKStub,
    make_settings: Callable[..., Settings],
) -> None:
    first = LLMClient(make_settings())
    second = LLMClient(make_settings())
    _ = first.anthropic
    _ = second.anthropic
    assert len(stub_anthropic.instances) == 2


def test_sdk_client_is_constructed_once_and_cached(
    stub_anthropic: SDKStub,
    settings: Settings,
) -> None:
    client = LLMClient(settings)
    assert client.anthropic is client.anthropic
    assert len(stub_anthropic.instances) == 1


def test_sdk_client_disables_its_own_retry_budget(
    stub_anthropic: SDKStub,
    settings: Settings,
) -> None:
    client = LLMClient(settings)
    _ = client.anthropic
    assert stub_anthropic.only_instance.init_kwargs["max_retries"] == 0


def test_anthropic_key_comes_from_settings_when_present(
    stub_anthropic: SDKStub,
    make_settings: Callable[..., Settings],
) -> None:
    client = LLMClient(make_settings(anthropic_api_key="from-settings"))
    _ = client.anthropic
    assert stub_anthropic.only_instance.init_kwargs["api_key"] == "from-settings"


# --------------------------------------------------------------------------------------
# Missing credentials
# --------------------------------------------------------------------------------------


def test_missing_anthropic_key_raises_without_constructing_a_client(
    stub_anthropic: SDKStub,
    make_settings: Callable[..., Settings],
) -> None:
    client = LLMClient(make_settings(anthropic_api_key=None))
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        _ = client.anthropic
    assert stub_anthropic.instances == []


def test_missing_openai_key_raises_without_constructing_a_client(
    make_settings: Callable[..., Settings],
) -> None:
    client = LLMClient(make_settings(openai_api_key=None))
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _ = client.openai


def test_environment_key_is_used_when_settings_have_none(
    monkeypatch: pytest.MonkeyPatch,
    stub_anthropic: SDKStub,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    client = LLMClient(build_settings())
    _ = client.anthropic
    assert stub_anthropic.only_instance.init_kwargs["api_key"] == "from-env"


async def test_complete_without_a_key_fails_immediately_and_is_not_retried(
    make_settings: Callable[..., Settings],
    no_sleep: list[float],
) -> None:
    client = LLMClient(make_settings(anthropic_api_key=None, max_retries=5))
    with pytest.raises(ValueError) as caught:
        await client.complete("hello")
    assert type(caught.value) is ValueError, "a missing key is not a retryable/reported LLMError"
    assert no_sleep == []


# --------------------------------------------------------------------------------------
# Provider dispatch
# --------------------------------------------------------------------------------------


async def test_anthropic_provider_sends_the_prompt_as_a_user_turn(
    stub_anthropic: SDKStub,
    stub_openai: SDKStub,
    make_settings: Callable[..., Settings],
) -> None:
    stub_anthropic.script("answered")
    client = LLMClient(make_settings(anthropic_api_key="k"))
    result = await client.complete("do the thing", system="be terse", model="my-model")

    assert result == "answered"
    request = stub_anthropic.last_request
    assert request["model"] == "my-model"
    assert request["system"] == "be terse"
    assert request["messages"] == [{"role": "user", "content": "do the thing"}]
    assert request["max_tokens"] == client.settings.max_tokens
    assert request["timeout"] == client.settings.request_timeout
    assert stub_openai.instances == []


async def test_default_system_prompt_is_substituted_for_an_empty_system(
    stub_anthropic: SDKStub,
    settings: Settings,
) -> None:
    stub_anthropic.script("ok")
    await LLMClient(settings).complete("hi")
    assert stub_anthropic.last_request["system"] == DEFAULT_SYSTEM_PROMPT


async def test_default_model_is_used_when_none_is_passed(
    stub_anthropic: SDKStub,
    make_settings: Callable[..., Settings],
) -> None:
    stub_anthropic.script("ok")
    await LLMClient(make_settings(model="fallback-model")).complete("hi")
    assert stub_anthropic.last_request["model"] == "fallback-model"


async def test_openai_provider_sends_a_system_message_and_a_user_message(
    stub_openai: SDKStub,
    stub_anthropic: SDKStub,
    make_settings: Callable[..., Settings],
) -> None:
    stub_openai.script("answered")
    client = LLMClient(make_settings(provider="openai", openai_api_key="k"))
    result = await client.complete("do it", system="be terse", model="gpt-x")

    assert result == "answered"
    request = stub_openai.last_request
    assert request["model"] == "gpt-x"
    assert request["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "do it"},
    ]
    assert stub_anthropic.instances == []


async def test_ollama_provider_routes_to_the_local_base_url(
    stub_openai: SDKStub,
    make_settings: Callable[..., Settings],
) -> None:
    stub_openai.script("local answer")
    settings = make_settings(
        provider="ollama", ollama_base_url="http://127.0.0.1:9999", openai_api_key="k"
    )

    assert await LLMClient(settings).complete("hi", model="llama3") == "local answer"

    assert len(stub_openai.instances) == 1, "ollama must use its own client, not the OpenAI one"
    assert stub_openai.only_instance.init_kwargs["base_url"] == "http://127.0.0.1:9999"
    assert stub_openai.last_request["model"] == "llama3"


async def test_unknown_provider_raises_llm_error(make_settings: Callable[..., Settings]) -> None:
    broken = make_settings().model_copy(update={"provider": "hal9000"})
    with pytest.raises(LLMError, match="Unsupported provider"):
        await LLMClient(broken).complete("hi")


def test_llm_error_is_a_value_error_so_existing_handlers_still_work() -> None:
    assert issubclass(LLMError, ValueError)


def test_settings_provider_round_trips_through_the_enum(settings: Settings) -> None:
    assert Provider(settings.provider) is Provider.ANTHROPIC


# --------------------------------------------------------------------------------------
# Anthropic content blocks
# --------------------------------------------------------------------------------------


async def test_text_after_a_thinking_block_is_still_returned(
    stub_anthropic: SDKStub,
    settings: Settings,
) -> None:
    # Regression: reading content[0].text made every extended-thinking reply a failure.
    stub_anthropic.script([FakeBlock("thinking", "let me think"), FakeBlock("text", "the answer")])
    assert await LLMClient(settings).complete("hi") == "the answer"


async def test_all_text_blocks_are_concatenated(
    stub_anthropic: SDKStub,
    settings: Settings,
) -> None:
    stub_anthropic.script(
        [
            FakeBlock("thinking", "hmm"),
            FakeBlock("text", "part one "),
            FakeBlock("tool_use"),
            FakeBlock("text", "part two"),
        ]
    )
    assert await LLMClient(settings).complete("hi") == "part one part two"


async def test_empty_text_blocks_are_skipped(
    stub_anthropic: SDKStub,
    settings: Settings,
) -> None:
    stub_anthropic.script([FakeBlock("text", ""), FakeBlock("text", "only this")])
    assert await LLMClient(settings).complete("hi") == "only this"


async def test_response_without_any_text_block_raises_llm_error(
    stub_anthropic: SDKStub,
    settings: Settings,
) -> None:
    stub_anthropic.script([FakeBlock("thinking", "no answer for you")])
    with pytest.raises(LLMError, match="no text block"):
        await LLMClient(settings).complete("hi")


async def test_empty_content_list_raises_llm_error(
    stub_anthropic: SDKStub,
    settings: Settings,
) -> None:
    stub_anthropic.script([])
    with pytest.raises(LLMError, match="no text block"):
        await LLMClient(settings).complete("hi")


# --------------------------------------------------------------------------------------
# OpenAI response validation
# --------------------------------------------------------------------------------------


async def test_openai_response_without_choices_raises_llm_error(
    stub_openai: SDKStub,
    make_settings: Callable[..., Settings],
) -> None:
    stub_openai.script(_Namespace(choices=[]))
    with pytest.raises(LLMError, match="no choices"):
        await LLMClient(make_settings(provider="openai")).complete("hi")


async def test_openai_response_with_empty_content_raises_llm_error(
    stub_openai: SDKStub,
    make_settings: Callable[..., Settings],
) -> None:
    stub_openai.script(None)
    with pytest.raises(LLMError, match="no message content"):
        await LLMClient(make_settings(provider="openai")).complete("hi")


# --------------------------------------------------------------------------------------
# Retry policy
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [rate_limited, internal_error, connection_error, timed_out, lambda: untyped_status_error(503)],
    ids=["429", "500", "connection", "timeout", "503-without-sdk-type"],
)
async def test_transient_failure_is_retried_until_the_budget_is_spent(
    failure: Callable[[], BaseException],
    stub_anthropic: SDKStub,
    make_settings: Callable[..., Settings],
    no_sleep: list[float],
) -> None:
    stub_anthropic.script(failure(), failure(), failure())
    client = LLMClient(make_settings(max_retries=3))

    with pytest.raises(LLMError, match="after 3 attempt"):
        await client.complete("hi")

    assert len(stub_anthropic.requests) == 3
    assert len(no_sleep) == 2, "one backoff between attempts, none after the final failure"


async def test_retry_recovers_when_a_later_attempt_succeeds(
    stub_anthropic: SDKStub,
    make_settings: Callable[..., Settings],
    no_sleep: list[float],
) -> None:
    stub_anthropic.script(rate_limited(), "second time lucky")
    client = LLMClient(make_settings(max_retries=3))

    assert await client.complete("hi") == "second time lucky"
    assert len(stub_anthropic.requests) == 2
    assert len(no_sleep) == 1


async def test_openai_transient_failure_is_retried_too(
    stub_openai: SDKStub,
    make_settings: Callable[..., Settings],
    no_sleep: list[float],
) -> None:
    failure = openai.RateLimitError("rate limited", response=_response(429), body=None)
    stub_openai.script(failure, failure)
    client = LLMClient(make_settings(provider="openai", max_retries=2))

    with pytest.raises(LLMError, match="after 2 attempt"):
        await client.complete("hi")
    assert len(stub_openai.requests) == 2


async def test_single_retry_setting_means_exactly_one_attempt(
    stub_anthropic: SDKStub,
    make_settings: Callable[..., Settings],
    no_sleep: list[float],
) -> None:
    stub_anthropic.script(rate_limited())
    client = LLMClient(make_settings(max_retries=1))

    with pytest.raises(anthropic.RateLimitError):
        await client.complete("hi")
    assert len(stub_anthropic.requests) == 1
    assert no_sleep == []


async def test_zero_retries_setting_also_means_one_attempt(
    stub_anthropic: SDKStub,
    make_settings: Callable[..., Settings],
    no_sleep: list[float],
) -> None:
    stub_anthropic.script(rate_limited())
    client = LLMClient(make_settings(max_retries=0))

    with pytest.raises(anthropic.RateLimitError):
        await client.complete("hi")
    assert len(stub_anthropic.requests) == 1


async def test_non_transient_value_error_is_raised_immediately(
    stub_anthropic: SDKStub,
    make_settings: Callable[..., Settings],
    no_sleep: list[float],
) -> None:
    stub_anthropic.script(ValueError("bad request, retrying will not help"))
    client = LLMClient(make_settings(max_retries=4))

    with pytest.raises(ValueError, match="will not help"):
        await client.complete("hi")
    assert len(stub_anthropic.requests) == 1
    assert no_sleep == []


async def test_client_side_error_is_not_retried(
    stub_anthropic: SDKStub,
    make_settings: Callable[..., Settings],
    no_sleep: list[float],
) -> None:
    rejection = anthropic.BadRequestError("nope", response=_response(400), body=None)
    stub_anthropic.script(rejection)
    client = LLMClient(make_settings(max_retries=4))

    with pytest.raises(anthropic.BadRequestError):
        await client.complete("hi")
    assert len(stub_anthropic.requests) == 1
    assert no_sleep == []


async def test_each_call_gets_its_own_retry_budget(
    stub_anthropic: SDKStub,
    make_settings: Callable[..., Settings],
    no_sleep: list[float],
) -> None:
    stub_anthropic.script(rate_limited())
    client = LLMClient(make_settings(max_retries=3))

    for _ in range(2):
        with pytest.raises(LLMError, match="after 3 attempt"):
            await client.complete("hi")

    assert len(stub_anthropic.requests) == 6
    assert len(no_sleep) == 4, "the budget is per call, not per client"


# --------------------------------------------------------------------------------------
# File helpers
# --------------------------------------------------------------------------------------


async def test_read_file_returns_the_contents(tmp_path: Path, settings: Settings) -> None:
    target = tmp_path / "note.txt"
    target.write_text("hello\nworld\n")
    assert await LLMClient(settings).read_file(target) == "hello\nworld\n"


async def test_read_file_accepts_a_string_path(tmp_path: Path, settings: Settings) -> None:
    target = tmp_path / "note.txt"
    target.write_text("text")
    assert await LLMClient(settings).read_file(str(target)) == "text"


async def test_read_file_raises_file_not_found_with_the_path(
    tmp_path: Path, settings: Settings
) -> None:
    with pytest.raises(FileNotFoundError, match=r"nope\.txt"):
        await LLMClient(settings).read_file(tmp_path / "nope.txt")


async def test_write_file_creates_missing_parent_directories(
    tmp_path: Path, settings: Settings
) -> None:
    target = tmp_path / "deep" / "deeper" / "new.py"
    await LLMClient(settings).write_file(target, "print('hi')\n")
    assert target.read_text() == "print('hi')\n"


async def test_write_file_overwrites_existing_content(tmp_path: Path, settings: Settings) -> None:
    target = tmp_path / "exists.py"
    target.write_text("old")
    await LLMClient(settings).write_file(target, "new")
    assert target.read_text() == "new"


async def test_write_then_read_round_trips_unicode(tmp_path: Path, settings: Settings) -> None:
    target = tmp_path / "unicode.txt"
    await LLMClient(settings).write_file(target, "héllo → wörld\n")
    assert await LLMClient(settings).read_file(target) == "héllo → wörld\n"


def test_list_files_returns_sorted_relative_posix_paths(
    tmp_path: Path, settings: Settings
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "b.py").write_text("")
    (tmp_path / "pkg" / "a.py").write_text("")
    (tmp_path / "top.md").write_text("")
    assert LLMClient(settings).list_files(tmp_path) == ["pkg/a.py", "pkg/b.py", "top.md"]


@pytest.mark.parametrize(
    "ignored", [".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"]
)
def test_list_files_skips_ignored_directories(
    ignored: str, tmp_path: Path, settings: Settings
) -> None:
    directory = tmp_path / ignored
    directory.mkdir()
    (directory / "junk.py").write_text("")
    (tmp_path / "keep.py").write_text("")
    assert LLMClient(settings).list_files(tmp_path) == ["keep.py"]


def test_list_files_excludes_bytecode_everywhere(
    tmp_path: Path, settings: Settings
) -> None:
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-310.pyc").write_bytes(b"\x00")
    (cache / "module.py").write_text("")
    (tmp_path / "loose.pyc").write_bytes(b"\x00")
    (tmp_path / "keep.py").write_text("")
    assert LLMClient(settings).list_files(tmp_path) == ["keep.py"]


def test_list_files_of_a_missing_directory_is_empty(tmp_path: Path, settings: Settings) -> None:
    assert LLMClient(settings).list_files(tmp_path / "does-not-exist") == []


def test_list_files_of_an_empty_directory_is_empty(tmp_path: Path, settings: Settings) -> None:
    assert LLMClient(settings).list_files(tmp_path) == []


def test_list_files_accepts_a_string_path(tmp_path: Path, settings: Settings) -> None:
    (tmp_path / "a.py").write_text("")
    assert LLMClient(settings).list_files(str(tmp_path)) == ["a.py"]
