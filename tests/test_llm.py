"""Provider transport: settings resolution, retries and response typing.

No SDK client is ever constructed here. The private ``_anthropic`` / ``_openai``
attributes are handed duck-typed fakes, so these tests exercise the real extraction,
retry and error-translation code without a network call or an API key.

The first two tests are the most important in the suite: the old constructor bound the
settings singleton as a parameter default, so ``LLMClient(settings=None)`` left
``self.settings`` as ``None`` and every agent's default constructor raised
``AttributeError`` on the following line.
"""

from __future__ import annotations

import asyncio
import socket
from types import SimpleNamespace
from typing import Any, cast

import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.agents.planner import PlannerAgent

# ``TesterAgent`` is imported as ``Grader``: a module-level name starting with ``Test``
# makes pytest try to collect the class as a test class and warn about its ``__init__``.
from furrow.agents.tester import TesterAgent as Grader
from furrow.agents.worker import WorkerAgent
from furrow.config import Provider, Settings, TaskModel
from furrow.llm import (
    DEFAULT_SYSTEM_PROMPT,
    TRANSIENT_ERROR_NAMES,
    LLMClient,
    LLMError,
    LLMResponseError,
    LLMTruncatedError,
    is_transient_error,
)

from .conftest import SettingsFactory, run_coroutine


class RateLimitError(Exception):
    """Stand-in for the provider SDK's rate-limit exception."""


class RateLimitSubclassError(RateLimitError):
    """A provider-specific subclass, which only the MRO walk can recognise."""


class FakeAnthropicSDK:
    """Duck-typed stand-in for ``anthropic.AsyncAnthropic``."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self.closed = False
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def close(self) -> None:
        self.closed = True


class FakeOpenAISDK:
    """Duck-typed stand-in for ``openai.AsyncOpenAI``."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self.closed = False
        self.completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=self.completions)

    async def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def close(self) -> None:
        self.closed = True


def anthropic_message(
    text: str | None = "answer", stop_reason: str = "end_turn", leading: tuple[str, ...] = ()
) -> Any:
    """A response shaped like an Anthropic message with *leading* non-text blocks."""
    blocks: list[Any] = [SimpleNamespace(type=kind) for kind in leading]
    if text is not None:
        blocks.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(content=blocks, stop_reason=stop_reason)


def openai_completion(
    content: str | None = "answer",
    finish_reason: str = "stop",
    choices: bool = True,
) -> Any:
    """A response shaped like an OpenAI chat completion."""
    if not choices:
        return SimpleNamespace(choices=[])
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=message)])


def anthropic_client(make_settings: SettingsFactory, outcomes: list[Any], **overrides: Any):
    """An :class:`LLMClient` wired to a fake Anthropic SDK."""
    settings = make_settings(provider=Provider.ANTHROPIC, **overrides)
    client = LLMClient(settings=settings)
    sdk = FakeAnthropicSDK(outcomes)
    client._anthropic = cast(AsyncAnthropic, sdk)
    return client, sdk, settings


def openai_client(make_settings: SettingsFactory, outcomes: list[Any], **overrides: Any):
    """An :class:`LLMClient` wired to a fake OpenAI SDK."""
    settings = make_settings(provider=Provider.OPENAI, **overrides)
    client = LLMClient(settings=settings)
    sdk = FakeOpenAISDK(outcomes)
    client._openai = cast(AsyncOpenAI, sdk)
    return client, sdk, settings


# ---------------------------------------------------------- settings resolution


def test_client_resolves_real_settings_when_given_none() -> None:
    client = LLMClient(settings=None)
    assert isinstance(client.settings, Settings)


def test_client_resolves_settings_when_constructed_with_no_arguments() -> None:
    assert isinstance(LLMClient().settings, Settings)


def test_client_keeps_an_explicit_settings_object() -> None:
    settings = Settings(model="explicit")
    assert LLMClient(settings=settings).settings is settings


def test_planner_agent_constructed_without_arguments_resolves_settings() -> None:
    assert isinstance(PlannerAgent().settings, Settings)


def test_worker_agent_constructed_with_only_a_task_resolves_settings() -> None:
    task = TaskModel(id="1", description="x")
    assert isinstance(WorkerAgent(task).settings, Settings)


def test_tester_agent_constructed_without_arguments_resolves_settings() -> None:
    assert isinstance(Grader().settings, Settings)


# ---------------------------------------------------------- transient detection


def test_synthetic_rate_limit_error_is_transient() -> None:
    assert is_transient_error(RateLimitError("slow down")) is True


def test_subclass_of_a_transient_error_is_transient() -> None:
    assert is_transient_error(RateLimitSubclassError("slow down")) is True


def test_value_error_is_not_transient() -> None:
    assert is_transient_error(ValueError("bad")) is False


def test_timeout_error_is_transient() -> None:
    assert is_transient_error(TimeoutError()) is True


def test_our_own_error_taxonomy_is_never_retried() -> None:
    assert is_transient_error(LLMError("nope")) is False
    assert is_transient_error(LLMResponseError("nope")) is False
    assert is_transient_error(LLMTruncatedError("nope")) is False


def test_documented_transient_names_are_reported() -> None:
    assert {"RateLimitError", "APITimeoutError", "InternalServerError"} <= TRANSIENT_ERROR_NAMES
    assert "AuthenticationError" not in TRANSIENT_ERROR_NAMES


# -------------------------------------------------------------------- retries


def test_zero_retries_means_exactly_one_provider_call(make_settings: SettingsFactory) -> None:
    client, sdk, _ = anthropic_client(
        make_settings, [RateLimitError("slow down")], llm_max_retries=0
    )
    with pytest.raises(LLMError):
        run_coroutine(client.complete("hello"))
    assert len(sdk.requests) == 1


def test_transient_failure_surfaces_as_an_llm_error_with_its_cause(
    make_settings: SettingsFactory,
) -> None:
    client, _, _ = anthropic_client(make_settings, [RateLimitError("slow down")], llm_max_retries=0)
    with pytest.raises(LLMError) as excinfo:
        run_coroutine(client.complete("hello"))
    assert isinstance(excinfo.value.__cause__, RateLimitError)
    assert "RateLimitError" in str(excinfo.value)


def test_transient_failures_are_retried_then_reported(make_settings: SettingsFactory) -> None:
    client, sdk, _ = anthropic_client(
        make_settings,
        [RateLimitError("one"), RateLimitError("two"), anthropic_message("third try")],
        llm_max_retries=2,
    )
    assert run_coroutine(client.complete("hello")) == "third try"
    assert len(sdk.requests) == 3


def test_a_timeout_becomes_an_llm_error(make_settings: SettingsFactory) -> None:
    class SlowAnthropicSDK(FakeAnthropicSDK):
        async def _create(self, **kwargs: Any) -> Any:
            await asyncio.sleep(5)
            raise AssertionError("the timeout should have fired first")

    client, _, settings = anthropic_client(make_settings, [], llm_timeout=0.01, llm_max_retries=0)
    client._anthropic = cast(AsyncAnthropic, SlowAnthropicSDK([]))
    with pytest.raises(LLMError, match="timed out after 0.01s"):
        run_coroutine(client.complete("hello"))


def test_a_permanent_provider_error_is_not_retried(make_settings: SettingsFactory) -> None:
    client, sdk, _ = anthropic_client(
        make_settings, [PermissionError("nope"), PermissionError("nope")], llm_max_retries=3
    )
    with pytest.raises(LLMError):
        run_coroutine(client.complete("hello"))
    assert len(sdk.requests) == 1


# --------------------------------------------------------- anthropic responses


def test_anthropic_returns_the_text_block(make_settings: SettingsFactory) -> None:
    client, _, _ = anthropic_client(make_settings, [anthropic_message("hello")])
    assert run_coroutine(client.complete("hi")) == "hello"


def test_anthropic_skips_a_leading_thinking_block(make_settings: SettingsFactory) -> None:
    client, _, _ = anthropic_client(
        make_settings, [anthropic_message("answer", leading=("thinking",))]
    )
    assert run_coroutine(client.complete("hi")) == "answer"


def test_anthropic_skips_a_leading_tool_use_block(make_settings: SettingsFactory) -> None:
    client, _, _ = anthropic_client(
        make_settings, [anthropic_message("answer", leading=("tool_use",))]
    )
    assert run_coroutine(client.complete("hi")) == "answer"


def test_anthropic_concatenates_every_text_block(make_settings: SettingsFactory) -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking"),
            SimpleNamespace(type="text", text="part one "),
            SimpleNamespace(type="text", text="part two"),
        ],
        stop_reason="end_turn",
    )
    client, _, _ = anthropic_client(make_settings, [response])
    assert run_coroutine(client.complete("hi")) == "part one part two"


def test_anthropic_empty_content_raises_a_response_error(make_settings: SettingsFactory) -> None:
    client, _, _ = anthropic_client(make_settings, [anthropic_message(text=None)])
    with pytest.raises(LLMResponseError, match="no text content"):
        run_coroutine(client.complete("hi"))


def test_anthropic_no_text_blocks_raises_a_response_error(make_settings: SettingsFactory) -> None:
    response = SimpleNamespace(
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="tool_use")],
        stop_reason="end_turn",
    )
    client, _, _ = anthropic_client(make_settings, [response])
    with pytest.raises(LLMResponseError):
        run_coroutine(client.complete("hi"))


def test_anthropic_max_tokens_stop_reason_raises_truncated(
    make_settings: SettingsFactory,
) -> None:
    client, _, _ = anthropic_client(
        make_settings,
        [anthropic_message("half a do", stop_reason="max_tokens")],
        llm_max_tokens=4096,
    )
    with pytest.raises(LLMTruncatedError, match="4096-token output limit"):
        run_coroutine(client.complete("hi", model="m"))


def test_truncated_is_not_a_response_error() -> None:
    assert issubclass(LLMTruncatedError, LLMError)
    assert not issubclass(LLMTruncatedError, LLMResponseError)


# ------------------------------------------------------------ openai responses


def test_openai_returns_the_message_content(make_settings: SettingsFactory) -> None:
    client, _, _ = openai_client(make_settings, [openai_completion("hello")])
    assert run_coroutine(client.complete("hi")) == "hello"


def test_openai_none_content_raises_a_response_error(make_settings: SettingsFactory) -> None:
    client, _, _ = openai_client(make_settings, [openai_completion(content=None)])
    with pytest.raises(LLMResponseError, match="no message content"):
        run_coroutine(client.complete("hi"))


def test_openai_empty_content_raises_a_response_error(make_settings: SettingsFactory) -> None:
    client, _, _ = openai_client(make_settings, [openai_completion(content="")])
    with pytest.raises(LLMResponseError):
        run_coroutine(client.complete("hi"))


def test_openai_empty_choices_raise_a_response_error(make_settings: SettingsFactory) -> None:
    client, _, _ = openai_client(make_settings, [openai_completion(choices=False)])
    with pytest.raises(LLMResponseError, match="no choices"):
        run_coroutine(client.complete("hi"))


def test_openai_length_finish_reason_raises_truncated(make_settings: SettingsFactory) -> None:
    client, _, _ = openai_client(
        make_settings, [openai_completion(finish_reason="length")], llm_max_tokens=4096
    )
    with pytest.raises(LLMTruncatedError, match="4096-token output limit"):
        run_coroutine(client.complete("hi", model="m"))


# ------------------------------------------------------------- request shaping


def test_anthropic_receives_the_configured_max_tokens(make_settings: SettingsFactory) -> None:
    client, sdk, settings = anthropic_client(
        make_settings, [anthropic_message("ok")], llm_max_tokens=1234
    )
    run_coroutine(client.complete("hi"))
    assert sdk.requests[0]["max_tokens"] == settings.llm_max_tokens == 1234


def test_openai_receives_the_configured_max_tokens(make_settings: SettingsFactory) -> None:
    client, sdk, settings = openai_client(
        make_settings, [openai_completion("ok")], llm_max_tokens=4321
    )
    run_coroutine(client.complete("hi"))
    assert sdk.requests[0]["max_tokens"] == settings.llm_max_tokens == 4321


def test_anthropic_falls_back_to_the_default_system_prompt(make_settings: SettingsFactory) -> None:
    client, sdk, _ = anthropic_client(make_settings, [anthropic_message("ok")])
    run_coroutine(client.complete("hi"))
    assert sdk.requests[0]["system"] == DEFAULT_SYSTEM_PROMPT


def test_anthropic_forwards_an_explicit_system_prompt(make_settings: SettingsFactory) -> None:
    client, sdk, _ = anthropic_client(make_settings, [anthropic_message("ok")])
    run_coroutine(client.complete("hi", system="be terse"))
    assert sdk.requests[0]["system"] == "be terse"


def test_anthropic_sends_the_prompt_as_a_user_message(make_settings: SettingsFactory) -> None:
    client, sdk, _ = anthropic_client(make_settings, [anthropic_message("ok")])
    run_coroutine(client.complete("build it"))
    assert sdk.requests[0]["messages"] == [{"role": "user", "content": "build it"}]


def test_openai_falls_back_to_the_default_system_prompt(make_settings: SettingsFactory) -> None:
    client, sdk, _ = openai_client(make_settings, [openai_completion("ok")])
    run_coroutine(client.complete("hi"))
    assert sdk.requests[0]["messages"][0] == {
        "role": "system",
        "content": DEFAULT_SYSTEM_PROMPT,
    }


def test_default_model_is_the_configured_model(make_settings: SettingsFactory) -> None:
    client, sdk, settings = anthropic_client(make_settings, [anthropic_message("ok")], model="m1")
    run_coroutine(client.complete("hi"))
    assert sdk.requests[0]["model"] == settings.model == "m1"


def test_explicit_model_overrides_the_default(make_settings: SettingsFactory) -> None:
    client, sdk, _ = anthropic_client(make_settings, [anthropic_message("ok")])
    run_coroutine(client.complete("hi", model="other"))
    assert sdk.requests[0]["model"] == "other"


# -------------------------------------------------------------- configuration


def test_missing_anthropic_key_raises_an_error_naming_the_variable() -> None:
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY is not set"):
        run_coroutine(client.complete("hi"))


def test_missing_openai_key_raises_an_error_naming_the_variable() -> None:
    client = LLMClient(settings=Settings(provider=Provider.OPENAI))
    with pytest.raises(LLMError, match="OPENAI_API_KEY is not set"):
        run_coroutine(client.complete("hi"))


def test_an_unknown_provider_is_reported_as_an_llm_error(
    make_settings: SettingsFactory,
) -> None:
    client = LLMClient(settings=make_settings().model_copy(update={"provider": "mystery"}))
    with pytest.raises(LLMError, match="Unsupported provider"):
        run_coroutine(client.complete("hi"))


# --------------------------------------------------------------------- offline


def test_the_offline_guard_refuses_a_real_connection() -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(RuntimeError, match="network connection"):
        probe.connect(("127.0.0.1", 9))


# ------------------------------------------------------------------ lifecycle


def test_aclose_is_a_noop_when_no_client_was_instantiated(
    make_settings: SettingsFactory,
) -> None:
    client = LLMClient(settings=make_settings())
    run_coroutine(client.aclose())


def test_aclose_leaves_a_client_that_was_never_instantiated(
    make_settings: SettingsFactory,
) -> None:
    client, anthropic_sdk, _ = anthropic_client(make_settings, [])
    run_coroutine(client.aclose())
    assert anthropic_sdk.closed is True
    assert client._anthropic is None
    assert client._openai is None


def test_aclose_releases_every_client_it_did_instantiate(make_settings: SettingsFactory) -> None:
    client, anthropic_sdk, _ = anthropic_client(make_settings, [])
    openai_sdk = FakeOpenAISDK([])
    client._openai = cast(AsyncOpenAI, openai_sdk)
    run_coroutine(client.aclose())
    assert anthropic_sdk.closed is True
    assert openai_sdk.closed is True
    assert client._openai is None


def test_aclose_reports_a_close_failure(make_settings: SettingsFactory) -> None:
    class FailingSDK(FakeAnthropicSDK):
        async def close(self) -> None:
            raise OSError("pool already gone")

    client, _, _ = anthropic_client(make_settings, [])
    client._anthropic = cast(AsyncAnthropic, FailingSDK([]))
    with pytest.raises(LLMError, match="failed to close"):
        run_coroutine(client.aclose())


def test_aclose_reports_a_close_failure_from_the_second_client(
    make_settings: SettingsFactory,
) -> None:
    class FailingSDK(FakeOpenAISDK):
        async def close(self) -> None:
            raise OSError("pool already gone")

    client, anthropic_sdk, _ = anthropic_client(make_settings, [])
    client._openai = cast(AsyncOpenAI, FailingSDK([]))
    with pytest.raises(LLMError, match="failed to close"):
        run_coroutine(client.aclose())
    assert anthropic_sdk.closed is True


def test_context_manager_closes_the_clients(make_settings: SettingsFactory) -> None:
    client, anthropic_sdk, _ = anthropic_client(make_settings, [])

    async def use() -> None:
        async with client:
            pass

    run_coroutine(use())
    assert anthropic_sdk.closed is True


def test_anthropic_sdk_is_never_constructed_during_construction(
    make_settings: SettingsFactory,
) -> None:
    client = LLMClient(settings=make_settings())
    assert client._anthropic is None
    assert client._openai is None


def test_anthropic_property_returns_the_injected_client(make_settings: SettingsFactory) -> None:
    client, sdk, _ = anthropic_client(make_settings, [])
    assert client.anthropic is sdk


def test_openai_property_returns_the_injected_client(make_settings: SettingsFactory) -> None:
    client, sdk, _ = openai_client(make_settings, [])
    assert client.openai is sdk
