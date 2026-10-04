"""Offline tests for `furrow.llm`: parsing, provider wiring, tool hand-off and retries.

No test in this module touches a provider. The Anthropic / OpenAI request methods are
replaced with canned responses, and an autouse guard makes any real HTTP request raise.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import httpx
import httpx2
import pytest
from anthropic import (
    APIConnectionError as AnthropicConnectionError,
)
from anthropic import (
    AuthenticationError as AnthropicAuthError,
)
from anthropic import (
    RateLimitError as AnthropicRateLimitError,
)
from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock, Usage
from openai.types.chat import ChatCompletion, ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import Function
from openai.types.completion_usage import CompletionUsage

from furrow.agents.tools import ToolContext, ToolRegistry
from furrow.config import Provider, Settings, reload_settings
from furrow.llm import (
    LLMClient,
    LLMConfigurationError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    ToolCall,
    ToolResult,
    ToolSpec,
)

TRUNCATION = "\n[warning: response truncated at max_tokens]"


# --------------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any real HTTP request impossible, so a leak fails loudly instead of hanging."""

    def blocked(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("network access is blocked in the furrow test suite")

    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked)
    # The provider SDKs ship their own httpx fork; block that transport too.
    monkeypatch.setattr(httpx2.Client, "send", blocked)
    monkeypatch.setattr(httpx2.AsyncClient, "send", blocked)


@pytest.fixture
def client(settings: Settings) -> LLMClient:
    """A real `LLMClient` on the fixture settings; its provider calls are always stubbed."""
    return LLMClient(settings)


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse tenacity's backoff to nothing.

    `wait_exponential_jitter` adds up to a full second of RANDOM jitter per attempt, which
    would make the retry tests slow and non-deterministic. The attempt counts and the retry
    predicate - the parts under test - stay real.
    """

    async def instant(seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instant)


# --------------------------------------------------------------------- builders
def text_block(text: str) -> TextBlock:
    return TextBlock(text=text, type="text")


def thinking_block() -> ThinkingBlock:
    return ThinkingBlock(thinking="let me think", signature="sig", type="thinking")


def tool_use_block(name: str, arguments: dict[str, Any], call_id: str = "tool_1") -> ToolUseBlock:
    return ToolUseBlock(id=call_id, name=name, input=arguments, type="tool_use")


def message(
    *blocks: Any,
    stop_reason: str = "end_turn",
    input_tokens: int = 3,
    output_tokens: int = 4,
) -> Message:
    return Message(
        id="msg_test",
        content=list(blocks),
        model="claude-test",
        role="assistant",
        stop_reason=cast("Any", stop_reason),
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def completion(
    text: str = "",
    *,
    finish_reason: str = "stop",
    prompt_tokens: int = 7,
    completion_tokens: int = 9,
    tool_calls: list[ChatCompletionMessageFunctionToolCall] | None = None,
) -> ChatCompletion:
    message_ = ChatCompletionMessage(
        content=text or None,
        tool_calls=cast("Any", tool_calls or None),
        role="assistant",
    )
    return ChatCompletion(
        id="chatcmpl_test",
        choices=[
            Choice(
                finish_reason=finish_reason,  # type: ignore[arg-type]
                index=0,
                message=message_,
            )
        ],
        created=0,
        model="gpt-test",
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def queue_replies(
    target: LLMClient,
    monkeypatch: pytest.MonkeyPatch,
    *replies: str,
) -> list[dict[str, Any]]:
    """Feed canned texts to `_anthropic_request`; return the list of captured payloads."""
    captured: list[dict[str, Any]] = []
    remaining = list(replies)

    async def fake(prompt: str, **payload: Any) -> Message:
        captured.append({"prompt": prompt, **payload})
        return message(text_block(remaining.pop(0) if remaining else ""))

    monkeypatch.setattr(target, "_anthropic_request", fake)
    return captured


def queue_openai_replies(
    target: LLMClient,
    monkeypatch: pytest.MonkeyPatch,
    *replies: ChatCompletion,
) -> list[dict[str, Any]]:
    """Feed canned chat completions to `_openai_request`; return the captured payloads."""
    captured: list[dict[str, Any]] = []
    remaining = list(replies)

    async def fake(prompt: str, **payload: Any) -> ChatCompletion:
        captured.append({"prompt": prompt, **payload})
        return remaining.pop(0) if remaining else completion()

    monkeypatch.setattr(target, "_openai_request", fake)
    return captured


def fast_retry(workspace: Any, **overrides: Any) -> Settings:
    """Settings whose retry loop is real but whose backoff is negligible."""
    defaults: dict[str, Any] = {
        "max_retries": 2,
        "retry_base_delay": 0.001,
        "request_timeout": 5.0,
    }
    return reload_settings(workspace=workspace, **{**defaults, **overrides})


def api_request() -> httpx2.Request:
    # The Anthropic SDK ships httpx2, so its error constructors demand httpx2 objects.
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def api_response(status: int) -> httpx2.Response:
    return httpx2.Response(status, request=api_request(), headers={"request-id": "req_test"})


def registry(workspace: Any, settings: Settings) -> ToolRegistry:
    """A real sandboxed tool registry, used only to prove the tools moved out of the LLM."""
    return ToolRegistry(ToolContext(workspace=workspace, settings=settings))


# ------------------------------------------------------------------- networking
async def test_the_network_guard_is_live() -> None:
    # Proves the autouse guard is installed, so "no network" is enforced, not assumed.
    with pytest.raises(RuntimeError, match="network access is blocked"):
        async with httpx.AsyncClient() as http_client:
            await http_client.get("http://127.0.0.1:1/v1/models")


# ----------------------------------------------------------------- error types
def test_llm_error_is_a_runtime_error() -> None:
    assert issubclass(LLMError, RuntimeError)


@pytest.mark.parametrize(
    "error", [LLMConfigurationError, LLMResponseError, LLMRateLimitError]
)
def test_specific_errors_subclass_llm_error(error: type[LLMError]) -> None:
    assert issubclass(error, LLMError)


def test_llm_errors_are_distinct_siblings() -> None:
    assert not issubclass(LLMResponseError, LLMRateLimitError)
    assert not issubclass(LLMRateLimitError, LLMConfigurationError)
    assert not issubclass(LLMConfigurationError, LLMResponseError)


def test_llm_error_exports_are_complete() -> None:
    import furrow.llm as module

    assert set(module.__all__) == {
        "LLMClient",
        "LLMConfigurationError",
        "LLMError",
        "LLMResponseError",
        "LLMRateLimitError",
        "ToolCall",
        "ToolResult",
        "ToolSpec",
    }


# --------------------------------------------------------------- tool contracts
def test_tool_spec_is_a_total_typed_dict() -> None:
    assert set(ToolSpec.__required_keys__) == {"name", "description", "parameters"}
    assert set(ToolSpec.__annotations__) == {"name", "description", "parameters"}


def test_tool_spec_holds_a_plain_dict() -> None:
    spec: ToolSpec = {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {}},
    }
    assert spec["name"] == "read_file"


def test_tool_call_fields() -> None:
    call = ToolCall(id="t1", name="write_file", arguments={"path": "a.py", "content": "x"})
    assert (call.id, call.name, call.arguments["path"]) == ("t1", "write_file", "a.py")


def test_tool_call_is_slotted() -> None:
    call = ToolCall(id="t1", name="read_file", arguments={})
    with pytest.raises(AttributeError):
        call.arguments_text = "nope"  # type: ignore[attr-defined]


def test_tool_result_token_counters_default_to_zero() -> None:
    result = ToolResult(text="done", tool_calls=[], iterations=2, stop_reason="end_turn")
    assert result.input_tokens == 0
    assert result.output_tokens == 0


def test_tool_result_carries_the_full_log() -> None:
    log = [{"name": "read_file", "arguments": {"path": "a.py"}, "result": ""}]
    result = ToolResult(text="done", tool_calls=log, iterations=1, stop_reason="max_iterations")
    assert result.tool_calls[0]["result"] == ""
    assert result.stop_reason == "max_iterations"


@pytest.mark.parametrize(
    "name",
    ["read_file", "write_file", "edit_file", "list_files", "search", "run_command"],
)
def test_llm_client_exposes_no_file_or_shell_tools(name: str) -> None:
    # Regression: dead tool shims used to live on the client; they moved to the registry.
    assert not hasattr(LLMClient, name)


def test_the_moved_tools_are_reachable_through_the_registry(
    workspace: Any, settings: Settings
) -> None:
    available = {spec["name"] for spec in registry(workspace, settings).specs()}
    assert available == {
        "read_file",
        "write_file",
        "edit_file",
        "list_files",
        "search",
        "run_command",
    }


# ------------------------------------------------------------- complete / usage
async def test_complete_returns_the_model_text(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_replies(client, monkeypatch, "hello there")
    assert await client.complete("hi") == "hello there"


async def test_complete_passes_system_through_verbatim(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = queue_replies(client, monkeypatch, "a", "b")
    await client.complete("hi", system="be terse")
    assert captured[0]["system"] == "be terse"
    await client.complete("hi")
    assert captured[1]["system"] is None


async def test_complete_sends_the_whole_prompt(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = queue_replies(client, monkeypatch, "ok")
    await client.complete("Goal:\nship it")
    assert captured[0]["prompt"] == "Goal:\nship it"


async def test_complete_uses_the_configured_max_tokens(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: max_tokens was hardcoded to 4096 and silently truncated long answers.
    captured = queue_replies(client, monkeypatch, "ok")
    await client.complete("hi")
    assert client.settings.max_tokens == 8192
    assert captured[0]["max_tokens"] == client.settings.max_tokens


async def test_complete_honours_an_explicit_max_tokens(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = queue_replies(client, monkeypatch, "ok")
    await client.complete("hi", max_tokens=64, temperature=0.7)
    assert captured[0]["max_tokens"] == 64
    assert captured[0]["temperature"] == 0.7


async def test_complete_uses_the_configured_model(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = queue_replies(client, monkeypatch, "ok")
    await client.complete("hi")
    assert captured[0]["model"] == client.settings.model


async def test_complete_appends_a_truncation_warning(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake(prompt: str, **payload: Any) -> Message:
        return message(text_block("half a sen"), stop_reason="max_tokens")

    monkeypatch.setattr(client, "_anthropic_request", fake)
    assert await client.complete("hi") == f"half a sen{TRUNCATION}"


async def test_complete_does_not_warn_on_a_normal_stop(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_replies(client, monkeypatch, "all done")
    assert "warning" not in await client.complete("hi")


async def test_complete_reads_text_after_a_leading_non_text_block(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: the first block used to be assumed to be text -> AttributeError.
    async def fake(prompt: str, **payload: Any) -> Message:
        return message(
            thinking_block(),
            tool_use_block("read_file", {"path": "a.py"}),
            text_block("here"),
        )

    monkeypatch.setattr(client, "_anthropic_request", fake)
    assert await client.complete("hi") == "here"


async def test_complete_concatenates_every_text_block(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake(prompt: str, **payload: Any) -> Message:
        return message(text_block("one "), text_block("two"))

    monkeypatch.setattr(client, "_anthropic_request", fake)
    assert await client.complete("hi") == "one two"


async def test_complete_ignores_a_lone_tool_use_block(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake(prompt: str, **payload: Any) -> Message:
        return message(tool_use_block("list_files", {}))

    monkeypatch.setattr(client, "_anthropic_request", fake)
    assert await client.complete("hi") == ""


def openai_client(workspace: Any, **overrides: Any) -> LLMClient:
    # A key is supplied because `complete()` touches the lazy `openai` property even when
    # the request itself is stubbed; no socket is ever opened.
    defaults: dict[str, Any] = {"openai_api_key": "test-key"}
    return LLMClient(
        reload_settings(workspace=workspace, provider=Provider.OPENAI, **{**defaults, **overrides})
    )


async def test_complete_openai_reads_the_first_choice(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = openai_client(settings.workspace)
    queue_openai_replies(target, monkeypatch, completion("from openai"))
    assert await target.complete("hi") == "from openai"


async def test_complete_openai_appends_a_truncation_warning(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = openai_client(settings.workspace)
    queue_openai_replies(target, monkeypatch, completion("half", finish_reason="length"))
    assert await target.complete("hi") == f"half{TRUNCATION}"


async def test_complete_openai_tolerates_no_choices(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = openai_client(settings.workspace)
    empty = completion()
    empty.choices.clear()
    queue_openai_replies(target, monkeypatch, empty)
    assert await target.complete("hi") == ""


async def test_complete_openai_needs_no_tool_calls(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = openai_client(settings.workspace)
    queue_openai_replies(target, monkeypatch, completion("plain"))
    assert (await target.complete_with_tools("go", [], max_iterations=1)).stop_reason == "end_turn"


# ------------------------------------------------------------- regression: ollama
def ollama_client(workspace: Any, **overrides: Any) -> LLMClient:
    return LLMClient(
        reload_settings(workspace=workspace, provider=Provider.OLLAMA, **overrides)
    )


def test_ollama_is_reachable_and_not_an_unsupported_provider(settings: Settings) -> None:
    # Regression: Provider.OLLAMA raised "Unsupported provider" even though it is selectable.
    reachable = ollama_client(settings.workspace)
    assert reachable.settings.provider is Provider.OLLAMA
    assert str(reachable._ollama_client().base_url).rstrip("/") == "http://localhost:11434/v1"


def test_ollama_targets_the_configured_base_url(settings: Settings) -> None:
    custom = ollama_client(settings.workspace, ollama_base_url="http://gpu-box:9999/")
    assert str(custom._ollama_client().base_url).rstrip("/") == "http://gpu-box:9999/v1"


async def test_ollama_complete_returns_text_instead_of_raising(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    reachable = ollama_client(settings.workspace)
    queue_openai_replies(reachable, monkeypatch, completion("ollama says hi"))
    assert await reachable.complete("hi") == "ollama says hi"


def test_ollama_never_needs_an_api_key(settings: Settings) -> None:
    reachable = ollama_client(settings.workspace)
    assert reachable.settings.anthropic_api_key is None
    assert reachable.settings.openai_api_key is None
    assert reachable._ollama_client() is reachable._ollama_client()  # cached


def test_ollama_swaps_the_anthropic_default_model(settings: Settings) -> None:
    reachable = ollama_client(settings.workspace)
    assert reachable.settings.model.startswith("claude")
    assert reachable._resolve_model(None) == "llama3.1"


def test_ollama_keeps_an_explicit_ollama_model(settings: Settings) -> None:
    reachable = ollama_client(settings.workspace, model="qwen2.5-coder")
    assert reachable._resolve_model(None) == "qwen2.5-coder"
    assert reachable._resolve_model("mistral") == "mistral"


# ------------------------------------------------------------- complete_json
async def test_complete_json_parses_bare_json(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_replies(client, monkeypatch, '{"tasks": [{"id": "1"}]}')
    assert await client.complete_json("plan") == {"tasks": [{"id": "1"}]}


async def test_complete_json_strips_a_json_fence(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_replies(client, monkeypatch, '```json\n{"passed": true}\n```')
    assert await client.complete_json("verdict") == {"passed": True}


async def test_complete_json_strips_an_unlabelled_fence(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_replies(client, monkeypatch, '```\n[1, 2, 3]\n```')
    assert await client.complete_json("list") == [1, 2, 3]


async def test_complete_json_tolerates_leading_and_trailing_prose(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_replies(client, monkeypatch, 'Sure! Here you go:\n{"ok": 1}\nHope that helps.')
    assert await client.complete_json("verdict") == {"ok": 1}


async def test_complete_json_parses_a_top_level_list(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_replies(client, monkeypatch, '["a", "b"]')
    assert await client.complete_json("list") == ["a", "b"]


async def test_complete_json_repairs_on_the_second_attempt(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = queue_replies(client, monkeypatch, "not json at all", '{"ok": true}')
    assert await client.complete_json("verdict") == {"ok": True}
    assert len(captured) == 2
    assert captured[0]["prompt"] == "verdict"
    repair = captured[1]["prompt"]
    assert repair.startswith("verdict")
    assert "Reply with JSON only." in repair


async def test_complete_json_raises_after_the_attempt_budget(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = queue_replies(client, monkeypatch, "nope", "still nope")
    with pytest.raises(LLMResponseError):
        await client.complete_json("verdict", max_attempts=2)
    assert len(captured) == 2


async def test_complete_json_with_one_attempt_does_not_repair(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = queue_replies(client, monkeypatch, "nope", '{"ok": true}')
    with pytest.raises(LLMResponseError):
        await client.complete_json("verdict", max_attempts=1)
    assert len(captured) == 1


async def test_complete_json_clamps_a_zero_attempt_budget(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = queue_replies(client, monkeypatch, "nope", '{"ok": true}')
    with pytest.raises(LLMResponseError):
        await client.complete_json("verdict", max_attempts=0)
    assert len(captured) == 1


async def test_complete_json_error_quotes_the_offending_output(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_replies(client, monkeypatch, "definitely not json")
    with pytest.raises(LLMResponseError, match="not valid JSON"):
        await client.complete_json("verdict", max_attempts=1)


# ------------------------------------------------------------ complete_with_tools
async def test_complete_with_tools_ends_the_turn_without_tool_calls(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_replies(client, monkeypatch, "no tools needed")
    result = await client.complete_with_tools("do it", [], max_iterations=1)
    assert result.stop_reason == "end_turn"
    assert result.text == "no tools needed"
    assert result.tool_calls == []
    assert result.iterations == 1


async def test_complete_with_tools_surfaces_every_requested_call(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake(prompt: str, **payload: Any) -> Message:
        return message(
            thinking_block(),
            tool_use_block("read_file", {"path": "a.py"}, "t1"),
            tool_use_block("list_files", {"path": "."}, "t2"),
        )

    monkeypatch.setattr(client, "_anthropic_request", fake)
    seen: list[ToolCall] = []

    async def on_tool_call(call: ToolCall) -> None:
        seen.append(call)

    result = await client.complete_with_tools(
        "do it", [], max_iterations=1, on_tool_call=on_tool_call
    )
    assert [call.name for call in seen] == ["read_file", "list_files"]
    assert seen[0].arguments == {"path": "a.py"}
    assert [entry["name"] for entry in result.tool_calls] == ["read_file", "list_files"]
    assert result.tool_calls[0]["result"] == ""
    assert result.stop_reason == "max_iterations"


async def test_complete_with_tools_respects_the_iteration_cap(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    async def fake(prompt: str, **payload: Any) -> Message:
        calls.append(1)
        return message(tool_use_block("read_file", {"path": "a.py"}))

    monkeypatch.setattr(client, "_anthropic_request", fake)
    result = await client.complete_with_tools("do it", [], max_iterations=3)
    assert result.iterations == 3
    assert len(calls) == 3
    assert result.stop_reason == "max_iterations"


async def test_complete_with_tools_defaults_the_cap_to_settings(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    capped = LLMClient(reload_settings(workspace=settings.workspace, max_tool_iterations=2))
    calls: list[int] = []

    async def fake(prompt: str, **payload: Any) -> Message:
        calls.append(1)
        return message(tool_use_block("read_file", {"path": "a.py"}))

    monkeypatch.setattr(capped, "_anthropic_request", fake)
    result = await capped.complete_with_tools("do it", [])
    assert result.iterations == 2
    assert len(calls) == 2


async def test_complete_with_tools_reports_errors_instead_of_raising(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake(prompt: str, **payload: Any) -> Message:
        raise LLMConfigurationError("no key")

    monkeypatch.setattr(client, "_anthropic_request", fake)
    result = await client.complete_with_tools("do it", [], max_iterations=1)
    assert result.stop_reason == "error"
    assert result.text.startswith("[error] LLMConfigurationError")
    assert result.tool_calls == []


async def test_complete_with_tools_forwards_the_tool_specs(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = queue_replies(client, monkeypatch, "done")
    specs: list[ToolSpec] = [
        {"name": "read_file", "description": "read", "parameters": {"type": "object"}}
    ]
    await client.complete_with_tools("do it", specs, max_iterations=1)
    assert captured[0]["tools"] == specs


# ------------------------------------------------------------------- usage
def test_usage_starts_at_zero(client: LLMClient) -> None:
    assert client.usage == {"input_tokens": 0, "output_tokens": 0, "requests": 0}


async def test_complete_accumulates_usage(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_replies(client, monkeypatch, "a", "b")
    await client.complete("hi")
    await client.complete("hi again")
    assert client.usage == {"input_tokens": 6, "output_tokens": 8, "requests": 2}


async def test_complete_json_accumulates_usage(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_replies(client, monkeypatch, '{"ok": true}')
    await client.complete_json("hi")
    assert client.usage["requests"] == 1
    assert client.usage["input_tokens"] == 3


async def test_tool_rounds_accumulate_usage(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake(prompt: str, **payload: Any) -> Message:
        return message(tool_use_block("read_file", {"path": "a.py"}))

    monkeypatch.setattr(client, "_anthropic_request", fake)
    result = await client.complete_with_tools("do it", [], max_iterations=3)
    assert client.usage["requests"] == 3
    assert client.usage["input_tokens"] == 9
    assert client.usage["output_tokens"] == 12
    assert result.input_tokens == 9
    assert result.output_tokens == 12


async def test_a_failed_round_is_not_counted_as_usage(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake(prompt: str, **payload: Any) -> Message:
        raise LLMConfigurationError("no key")

    monkeypatch.setattr(client, "_anthropic_request", fake)
    await client.complete_with_tools("do it", [], max_iterations=1)
    assert client.usage == {"input_tokens": 0, "output_tokens": 0, "requests": 0}


def test_usage_returns_a_copy(client: LLMClient) -> None:
    snapshot = client.usage
    snapshot["requests"] = 99
    assert client.usage["requests"] == 0


def test_reset_usage_zeroes_the_counters(client: LLMClient) -> None:
    client._record_usage(5, 7)
    client.reset_usage()
    assert client.usage == {"input_tokens": 0, "output_tokens": 0, "requests": 0}


# --------------------------------------------------------- configuration errors
def test_anthropic_client_requires_a_key(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    unkeyed = LLMClient(reload_settings(workspace=settings.workspace, anthropic_api_key=None))
    with pytest.raises(LLMConfigurationError, match="FURROW_ANTHROPIC_API_KEY"):
        _ = unkeyed.anthropic


def test_openai_client_requires_a_key(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    unkeyed = LLMClient(reload_settings(workspace=settings.workspace, openai_api_key=None))
    with pytest.raises(LLMConfigurationError, match="FURROW_OPENAI_API_KEY"):
        _ = unkeyed.openai


def test_anthropic_client_uses_the_bare_env_key(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "bare-key")
    keyed = LLMClient(reload_settings(workspace=settings.workspace, anthropic_api_key=None))
    assert keyed.anthropic is not None


async def test_complete_without_a_key_raises_a_configuration_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    unkeyed = LLMClient(reload_settings(workspace=settings.workspace, anthropic_api_key=None))
    with pytest.raises(LLMConfigurationError):
        await unkeyed.complete("hi")


# ------------------------------------------------------------------ retry policy
async def test_a_transient_error_is_retried_then_succeeds(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, no_backoff: None
) -> None:
    attempts: list[int] = []

    async def fake(prompt: str, **payload: Any) -> Message:
        attempts.append(1)
        if len(attempts) == 1:
            raise AnthropicConnectionError(request=api_request())
        return message(text_block("recovered"))

    retrying = LLMClient(fast_retry(settings.workspace))
    monkeypatch.setattr(retrying, "_anthropic_request", fake)
    assert await retrying.complete("hi") == "recovered"
    assert len(attempts) == 2


async def test_an_auth_error_is_not_retried(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, no_backoff: None
) -> None:
    attempts: list[int] = []

    async def fake(prompt: str, **payload: Any) -> Message:
        attempts.append(1)
        raise AnthropicAuthError("bad key", response=api_response(401), body=None)

    retrying = LLMClient(fast_retry(settings.workspace, max_retries=3))
    monkeypatch.setattr(retrying, "_anthropic_request", fake)
    with pytest.raises(AnthropicAuthError):
        await retrying.complete("hi")
    assert len(attempts) == 1


async def test_exhausted_rate_limit_retries_raise_llm_rate_limit_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, no_backoff: None
) -> None:
    attempts: list[int] = []

    async def fake(prompt: str, **payload: Any) -> Message:
        attempts.append(1)
        raise AnthropicRateLimitError("429", response=api_response(429), body=None)

    retrying = LLMClient(fast_retry(settings.workspace, max_retries=2))
    monkeypatch.setattr(retrying, "_anthropic_request", fake)
    with pytest.raises(LLMRateLimitError, match="Rate limited after retries"):
        await retrying.complete("hi")
    assert len(attempts) == 3


async def test_exhausted_transient_retries_surface_the_original_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, no_backoff: None
) -> None:
    attempts: list[int] = []

    async def fake(prompt: str, **payload: Any) -> Message:
        attempts.append(1)
        raise AnthropicConnectionError(request=api_request())

    retrying = LLMClient(fast_retry(settings.workspace, max_retries=1))
    monkeypatch.setattr(retrying, "_anthropic_request", fake)
    with pytest.raises(AnthropicConnectionError):
        await retrying.complete("hi")
    assert len(attempts) == 2


async def test_one_retry_budget_means_two_attempts(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, no_backoff: None
) -> None:
    attempts: list[int] = []

    async def fake(prompt: str, **payload: Any) -> Message:
        attempts.append(1)
        raise AnthropicConnectionError(request=api_request())

    once = LLMClient(fast_retry(settings.workspace, max_retries=1))
    monkeypatch.setattr(once, "_anthropic_request", fake)
    with pytest.raises(AnthropicConnectionError):
        await once.complete("hi")
    assert len(attempts) == 2


async def test_zero_retries_means_a_single_attempt(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, no_backoff: None
) -> None:
    # WAVE 5b: this test used to pin `max_retries=0` -> 2 attempts, because `_retry`
    # floored the budget at `max(1, max_retries) + 1`. That was a policy bug: the setting
    # is named "retries", so 0 must mean "no retry". It now asserts one attempt only.
    attempts: list[int] = []

    async def fake(prompt: str, **payload: Any) -> Message:
        attempts.append(1)
        raise AnthropicConnectionError(request=api_request())

    once = LLMClient(fast_retry(settings.workspace, max_retries=0))
    monkeypatch.setattr(once, "_anthropic_request", fake)
    with pytest.raises(AnthropicConnectionError):
        await once.complete("hi")
    assert len(attempts) == 1


async def test_three_retries_means_four_attempts(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, no_backoff: None
) -> None:
    # Guards the other end of the same arithmetic: max_retries=3 -> 1 + 3 = 4 attempts.
    attempts: list[int] = []

    async def fake(prompt: str, **payload: Any) -> Message:
        attempts.append(1)
        raise AnthropicConnectionError(request=api_request())

    retrying = LLMClient(fast_retry(settings.workspace, max_retries=3))
    monkeypatch.setattr(retrying, "_anthropic_request", fake)
    with pytest.raises(AnthropicConnectionError):
        await retrying.complete("hi")
    assert len(attempts) == 4


# ------------------------------------------------------- request payload wiring
async def test_anthropic_request_builds_the_expected_payload(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    keyed = LLMClient(
        fast_retry(settings.workspace, anthropic_api_key="test-key", max_tokens=4321)
    )
    captured: dict[str, Any] = {}

    async def fake_create(**payload: Any) -> Message:
        captured.update(payload)
        return message(text_block("ok"))

    monkeypatch.setattr(keyed.anthropic.messages, "create", fake_create)
    await keyed._anthropic_request(
        "prompt body",
        system="rules",
        model="claude-x",
        max_tokens=keyed.settings.max_tokens,
        temperature=0.5,
        tools=[{"name": "read_file", "description": "read", "parameters": {"type": "object"}}],
    )
    assert captured["model"] == "claude-x"
    assert captured["max_tokens"] == 4321
    assert captured["system"] == "rules"
    assert captured["temperature"] == 0.5
    assert captured["messages"] == [{"role": "user", "content": "prompt body"}]
    assert captured["tools"] == [
        {"name": "read_file", "description": "read", "input_schema": {"type": "object"}}
    ]


async def test_anthropic_request_omits_unset_parameters(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    keyed = LLMClient(fast_retry(settings.workspace, anthropic_api_key="test-key"))
    captured: dict[str, Any] = {}

    async def fake_create(**payload: Any) -> Message:
        captured.update(payload)
        return message(text_block("ok"))

    monkeypatch.setattr(keyed.anthropic.messages, "create", fake_create)
    await keyed._anthropic_request(
        "body", system="", model="claude-x", max_tokens=10, temperature=None
    )
    assert "system" not in captured
    assert "temperature" not in captured


async def test_openai_request_puts_system_first(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    keyed = openai_client(settings.workspace, openai_api_key="test-key")
    captured: dict[str, Any] = {}

    async def fake_create(**payload: Any) -> ChatCompletion:
        captured.update(payload)
        return completion("ok")

    monkeypatch.setattr(keyed.openai.chat.completions, "create", fake_create)
    await keyed._openai_request(
        "body",
        client=keyed.openai,
        system="rules",
        model="gpt-x",
        max_tokens=99,
        temperature=0.0,
    )
    assert captured["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "body"},
    ]
    assert captured["max_tokens"] == 99
    assert captured["model"] == "gpt-x"


async def test_openai_tool_arguments_are_decoded_from_json(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Guards the hand-off contract: arguments arrive as a JSON string and must be decoded.
    target = openai_client(settings.workspace)
    call = ChatCompletionMessageFunctionToolCall(
        id="c1",
        function=Function(name="run_command", arguments='{"command": "ls"}'),
        type="function",
    )
    queue_openai_replies(target, monkeypatch, completion("", tool_calls=[call]))
    result = await target.complete_with_tools("go", [], max_iterations=1)
    assert result.tool_calls[0]["arguments"] == {"command": "ls"}
