"""Behavioural tests for :mod:`furrow.llm`.

Every test here is offline. The SDK clients are real objects (so the real
request plumbing, ``messages`` resource and ``chat.completions`` resource are
exercised) but their ``create`` methods are replaced by a stub that serves a
queue of pre-built responses and records what was sent. Nothing resolves a
hostname, nothing needs an API key, and because the stubs raise plain
``AssertionError`` -- which is not one of tenacity's retryable exceptions --
no test ever sits through a backoff sleep.

Responses are built from the real SDK types (``anthropic.types.Message`` /
``TextBlock`` / ``ToolUseBlock`` / ``ThinkingBlock``, ``openai.types.chat.*``)
rather than hand-rolled stand-ins: the tool loop replays
``block.model_dump(exclude_none=True)`` and inspects ``isinstance(block,
ToolUseBlock)``, so a duck-typed fake would assert against a fiction.

Two facts drive the assertions:

* the tool loops hand the *same* ``messages`` list to every request and mutate
  it in place, so the stub ``deepcopy``s each request on arrival -- otherwise
  every recorded request would show the final conversation state;
* ``_retry_call`` only retries the six SDK rate-limit/connection/timeout
  exceptions, so ``LLMResponseError`` and tool failures surface immediately.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock
from anthropic.types import Usage as AnthropicUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from furrow import llm as llm_module
from furrow.config import Provider, Settings
from furrow.config import settings as config_settings
from furrow.errors import (
    ConfigurationError,
    LLMResponseError,
    MissingCredentialsError,
)
from furrow.llm import DEFAULT_SYSTEM_PROMPT, LLMClient, Usage
from furrow.tools import WorkspaceTools

DUMMY_KEY = "sk-ant-not-a-real-key"
DUMMY_OPENAI_KEY = "sk-openai-not-a-real-key"


# --------------------------------------------------------------------------
# fake SDK responses
# --------------------------------------------------------------------------


def text_block(text: str) -> TextBlock:
    """A real Anthropic text block."""
    return TextBlock(type="text", text=text)


def tool_use_block(block_id: str, name: str, arguments: dict[str, Any]) -> ToolUseBlock:
    """A real Anthropic tool_use block."""
    return ToolUseBlock(type="tool_use", id=block_id, name=name, input=arguments)


def thinking_block(thinking: str = "let me think", signature: str = "sig") -> ThinkingBlock:
    """A real Anthropic thinking block (no ``.text`` attribute)."""
    return ThinkingBlock(type="thinking", thinking=thinking, signature=signature)


def anthropic_response(
    blocks: list[Any],
    *,
    model: str = "claude-test-1",
    stop_reason: str | None = "end_turn",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> Message:
    """A complete Anthropic Messages response."""
    return Message(
        id="msg_test",
        content=blocks,
        model=model,
        role="assistant",
        stop_reason=stop_reason,
        stop_sequence=None,
        type="message",
        usage=AnthropicUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def openai_tool_call(call_id: str, name: str, arguments: str) -> ChatCompletionMessageToolCall:
    """A real OpenAI tool call. ``arguments`` stays the raw JSON string."""
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function={"name": name, "arguments": arguments},
    )


def openai_response(
    *,
    text: str | None = None,
    tool_calls: list[ChatCompletionMessageToolCall] | None = None,
    finish_reason: str | None = "stop",
    model: str = "gpt-test-1",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> ChatCompletion:
    """A complete OpenAI chat completion with one choice."""
    message = ChatCompletionMessage(role="assistant", content=text, tool_calls=tool_calls)
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[Choice(index=0, finish_reason=finish_reason, message=message)],
        created=0,
        model=model,
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class _CreateRecorder:
    """The body of a stubbed ``create`` method.

    Serves queued responses in order and records one deep copy of every
    request. Asking for more responses than were scripted is a test bug, so it
    raises :class:`AssertionError` rather than silently looping.
    """

    def __init__(self, responses: list[Any], label: str) -> None:
        self._responses = list(responses)
        self._label = label
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        # deepcopy is essential: both tool loops mutate the very `messages`
        # list they pass in, so a shallow record would be rewritten by the
        # following round.
        self.requests.append(deepcopy(kwargs))
        if not self._responses:
            raise AssertionError(
                f"unexpected {self._label} call #{len(self.requests)}: no scripted response left"
            )
        return self._responses.pop(0)


@asynccontextmanager
async def anthropic_client(
    cfg: Settings,
    responses: list[Any],
    *,
    tools: WorkspaceTools | None = None,
) -> AsyncIterator[tuple[LLMClient, list[dict[str, Any]]]]:
    """A client whose ``messages.create`` is stubbed; yields (client, requests)."""
    client = LLMClient(cfg, tools=tools)
    recorder = _CreateRecorder(responses, "messages.create")
    client.anthropic.messages.create = recorder.create  # type: ignore[method-assign]
    try:
        yield client, recorder.requests
    finally:
        await client.aclose()


@asynccontextmanager
async def openai_client(
    cfg: Settings,
    responses: list[Any],
    *,
    tools: WorkspaceTools | None = None,
) -> AsyncIterator[tuple[LLMClient, list[dict[str, Any]]]]:
    """A client whose ``chat.completions.create`` is stubbed."""
    client = LLMClient(cfg, tools=tools)
    recorder = _CreateRecorder(responses, "chat.completions.create")
    client.openai.chat.completions.create = recorder.create  # type: ignore[method-assign]
    try:
        yield client, recorder.requests
    finally:
        await client.aclose()


def anthropic_settings(make_settings: Callable[..., Settings], **extra: Any) -> Settings:
    """Settings for the Anthropic path with a dummy key (never used)."""
    return make_settings(provider="anthropic", anthropic_api_key=DUMMY_KEY, **extra)


def openai_settings(make_settings: Callable[..., Settings], **extra: Any) -> Settings:
    """Settings for the OpenAI path with a dummy key (never used)."""
    return make_settings(provider="openai", openai_api_key=DUMMY_OPENAI_KEY, **extra)


# --------------------------------------------------------------------------
# 1. settings resolution
# --------------------------------------------------------------------------


def test_client_with_no_arguments_uses_the_settings_singleton(agent_env: Path) -> None:
    """``LLMClient()`` must resolve the module singleton, never forward ``None``.

    A ``settings=None`` default that was passed straight through used to make
    the very next attribute access raise ``AttributeError: 'NoneType' object
    has no attribute 'planner_model'``.
    """
    client = LLMClient()

    assert client.settings is not None
    assert client.settings is config_settings
    # The attribute that the old crash named, still reachable through the client.
    assert client.settings.planner_model == config_settings.planner_model


def test_client_uses_the_settings_instance_it_was_given(
    make_settings: Callable[..., Settings],
) -> None:
    cfg = make_settings(model="explicit-model")

    client = LLMClient(cfg)

    assert client.settings is cfg
    assert client.settings.model == "explicit-model"


def test_client_keeps_its_tools_and_has_none_by_default(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    cfg = make_settings()
    tools = WorkspaceTools(tmp_path)

    assert LLMClient(cfg).tools is None
    assert LLMClient(cfg, tools=tools).tools is tools


# --------------------------------------------------------------------------
# 2. credentials
# --------------------------------------------------------------------------


def test_anthropic_without_a_key_raises_missing_credentials(
    agent_env: Path, make_settings: Callable[..., Settings]
) -> None:
    cfg = make_settings(provider="anthropic", anthropic_api_key=None)

    with pytest.raises(MissingCredentialsError) as excinfo:
        _ = LLMClient(cfg).anthropic

    assert isinstance(excinfo.value, ConfigurationError)
    assert excinfo.value.details["provider"] == Provider.ANTHROPIC.value


def test_openai_without_a_key_raises_missing_credentials(
    agent_env: Path, make_settings: Callable[..., Settings]
) -> None:
    cfg = make_settings(provider="openai", openai_api_key=None)

    with pytest.raises(MissingCredentialsError) as excinfo:
        _ = LLMClient(cfg).openai

    assert isinstance(excinfo.value, ConfigurationError)
    assert excinfo.value.details["provider"] == Provider.OPENAI.value


async def test_a_credential_failure_only_happens_when_the_client_is_built(
    agent_env: Path, make_settings: Callable[..., Settings]
) -> None:
    """Construction never raises; the key is only needed for a real call."""
    cfg = make_settings(provider="anthropic", anthropic_api_key=None)

    client = LLMClient(cfg)  # does not raise

    with pytest.raises(MissingCredentialsError):
        await client.complete("hi")


# --------------------------------------------------------------------------
# 3. Ollama through the OpenAI-compatible client
# --------------------------------------------------------------------------


async def test_ollama_uses_an_openai_compatible_client(
    agent_env: Path, make_settings: Callable[..., Settings]
) -> None:
    """Ollama must not fall through to "Unsupported provider"."""
    cfg = make_settings(
        provider="ollama",
        ollama_base_url="http://127.0.0.1:11434/v1",
        anthropic_api_key=None,
        openai_api_key=None,
    )
    response = openai_response(text="ollama says hi", model="qwen2.5-coder:14b")

    async with openai_client(cfg, [response]) as (client, requests):
        sdk_client = client.openai
        # httpx -- which the OpenAI SDK builds AsyncOpenAI on -- enforces a
        # trailing slash on ``base_url``, so the SDK stores ".../v1/" even though
        # the setting has none. Normalise both sides before comparing.
        assert str(sdk_client.base_url).rstrip("/") == cfg.ollama_base_url.rstrip("/")
        assert client.openai is sdk_client, "the ollama client must be cached, not rebuilt"

        result = await client.complete_detailed("hello")

    assert result.text == "ollama says hi"
    assert result.provider == Provider.OLLAMA.value
    assert result.model == "qwen2.5-coder:14b"
    assert len(requests) == 1
    assert [message["role"] for message in requests[0]["messages"]] == ["system", "user"]


# --------------------------------------------------------------------------
# 4. Anthropic text extraction
# --------------------------------------------------------------------------


async def test_anthropic_concatenates_text_and_skips_tool_and_thinking_blocks(
    make_settings: Callable[..., Settings],
) -> None:
    cfg = anthropic_settings(make_settings)
    response = anthropic_response(
        [
            thinking_block(),
            text_block("Hello, "),
            tool_use_block("toolu_1", "write_file", {"path": "a.txt", "content": "x"}),
            text_block("world."),
        ],
        stop_reason="tool_use",
        input_tokens=11,
        output_tokens=7,
    )

    async with anthropic_client(cfg, [response]) as (client, requests):
        result = await client.complete_detailed("hi")

    assert result.text == "Hello, world."
    assert result.truncated is False
    assert result.usage == Usage(input_tokens=11, output_tokens=7)
    assert result.tool_calls == []
    # model=None falls back to Settings.model; the request carried it through.
    assert requests[0]["model"] == cfg.model


async def test_anthropic_truncates_on_max_tokens(
    make_settings: Callable[..., Settings],
) -> None:
    cfg = anthropic_settings(make_settings)
    response = anthropic_response(
        [text_block("cut off here")],
        stop_reason="max_tokens",
    )

    async with anthropic_client(cfg, [response]) as (client, _requests):
        result = await client.complete_detailed("hi")

    assert result.text == "cut off here"
    assert result.truncated is True


async def test_anthropic_honours_an_explicit_model_override(
    make_settings: Callable[..., Settings],
) -> None:
    cfg = anthropic_settings(make_settings)
    response = anthropic_response([text_block("ok")])

    async with anthropic_client(cfg, [response]) as (client, requests):
        result = await client.complete_detailed("hi", model="claude-override", system="be brief")

    assert requests[0]["model"] == "claude-override"
    assert requests[0]["system"] == "be brief"
    assert result.model == "claude-test-1", "the response model wins over the requested one"


async def test_anthropic_defaults_an_empty_system_prompt(
    make_settings: Callable[..., Settings],
) -> None:
    cfg = anthropic_settings(make_settings)

    async with anthropic_client(cfg, [anthropic_response([text_block("ok")])]) as (
        client,
        requests,
    ):
        await client.complete("hi")

    assert requests[0]["system"] == DEFAULT_SYSTEM_PROMPT
    assert requests[0]["messages"] == [{"role": "user", "content": "hi"}]


async def test_anthropic_returns_only_the_text_from_complete(
    make_settings: Callable[..., Settings],
) -> None:
    cfg = anthropic_settings(make_settings)

    async with anthropic_client(cfg, [anthropic_response([text_block("just text")])]) as (
        client,
        _requests,
    ):
        assert await client.complete("hi") == "just text"


# --------------------------------------------------------------------------
# 5. Anthropic response with no text
# --------------------------------------------------------------------------


async def test_anthropic_raises_when_the_response_has_no_text_block(
    make_settings: Callable[..., Settings],
) -> None:
    cfg = anthropic_settings(make_settings)
    response = anthropic_response(
        [tool_use_block("toolu_1", "write_file", {"path": "a.txt", "content": "x"})],
        stop_reason="tool_use",
    )

    async with anthropic_client(cfg, [response]) as (client, _requests):
        with pytest.raises(LLMResponseError) as excinfo:
            await client.complete_detailed("hi")

    # Raised as-is by the extractor, not laundered through the retry wrapper.
    assert excinfo.value.message == "Anthropic response contained no text blocks"
    assert excinfo.value.details["model"] == "claude-test-1"
    assert excinfo.value.details["stop_reason"] == "tool_use"
    assert client.usage == Usage(), "a failed extraction must not bill tokens"


# --------------------------------------------------------------------------
# 6. OpenAI usage and truncation
# --------------------------------------------------------------------------


async def test_openai_reads_usage_and_flags_length_as_truncated(
    make_settings: Callable[..., Settings],
) -> None:
    cfg = openai_settings(make_settings)
    response = openai_response(
        text="a truncated answe",
        finish_reason="length",
        prompt_tokens=31,
        completion_tokens=17,
    )

    async with openai_client(cfg, [response]) as (client, _requests):
        result = await client.complete_detailed("hi")

    assert result.text == "a truncated answe"
    assert result.truncated is True
    assert result.usage == Usage(input_tokens=31, output_tokens=17)
    assert result.provider == Provider.OPENAI.value


async def test_openai_stop_finish_reason_is_not_truncated(
    make_settings: Callable[..., Settings],
) -> None:
    cfg = openai_settings(make_settings)

    async with openai_client(
        cfg, [openai_response(text="complete", finish_reason="stop")]
    ) as (client, _requests):
        result = await client.complete_detailed("hi")

    assert result.truncated is False


# --------------------------------------------------------------------------
# 7. Usage arithmetic and accumulation
# --------------------------------------------------------------------------


def test_usage_add_is_elementwise_and_total_is_the_sum() -> None:
    left = Usage(input_tokens=1, output_tokens=2)
    right = Usage(input_tokens=10, output_tokens=20)

    total = left + right

    assert total == Usage(input_tokens=11, output_tokens=22)
    assert total.total_tokens == 33
    assert total is not left and total is not right
    assert left == Usage(input_tokens=1, output_tokens=2), "operands must be untouched"
    assert right == Usage(input_tokens=10, output_tokens=20)


def test_usage_defaults_to_zero() -> None:
    empty = Usage()

    assert empty.input_tokens == 0
    assert empty.output_tokens == 0
    assert empty.total_tokens == 0


async def test_client_usage_accumulates_across_calls_and_hands_out_copies(
    make_settings: Callable[..., Settings],
) -> None:
    cfg = anthropic_settings(make_settings)
    responses = [
        anthropic_response([text_block("one")], input_tokens=5, output_tokens=7),
        anthropic_response([text_block("two")], input_tokens=11, output_tokens=3),
    ]

    async with anthropic_client(cfg, responses) as (client, _requests):
        assert client.usage == Usage()

        first = await client.complete_detailed("one")
        assert client.usage == Usage(input_tokens=5, output_tokens=7)

        second = await client.complete_detailed("two")
        assert client.usage == Usage(input_tokens=16, output_tokens=10)
        assert client.usage.total_tokens == 26

        # Each call reports only its own usage...
        assert first.usage == Usage(input_tokens=5, output_tokens=7)
        assert second.usage == Usage(input_tokens=11, output_tokens=3)

        # ...and .usage is a copy, so mutating it cannot corrupt the counter.
        snapshot = client.usage
        snapshot.input_tokens = 999
        assert client.usage == Usage(input_tokens=16, output_tokens=10)


async def test_reset_usage_zeroes_the_counter(make_settings: Callable[..., Settings]) -> None:
    cfg = anthropic_settings(make_settings)
    response = anthropic_response([text_block("one")], input_tokens=5, output_tokens=7)

    async with anthropic_client(cfg, [response]) as (client, _requests):
        await client.complete_detailed("one")
        assert client.usage != Usage()

        client.reset_usage()

        assert client.usage == Usage()
        client.reset_usage()
        assert client.usage == Usage(), "resetting twice is harmless"


# --------------------------------------------------------------------------
# 8. the Anthropic agentic tool loop
# --------------------------------------------------------------------------


async def test_anthropic_tool_loop_writes_the_file_then_answers(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    cfg = anthropic_settings(make_settings)
    tools = WorkspaceTools(tmp_path)
    payload = "print('hello')\n"
    responses = [
        anthropic_response(
            [tool_use_block("toolu_1", "write_file", {"path": "hello.py", "content": payload})],
            stop_reason="tool_use",
            input_tokens=20,
            output_tokens=15,
        ),
        anthropic_response(
            [text_block("Wrote hello.py.")],
            input_tokens=40,
            output_tokens=5,
        ),
    ]

    async with anthropic_client(cfg, responses, tools=tools) as (client, requests):
        # Tools come from the client; the call does not need to pass them again.
        result = await client.complete_with_tools("write hello.py", system="you are a worker")

        # The tool really ran: the file is on disk with the requested content.
        written = tmp_path / "hello.py"
        assert written.exists()
        assert written.read_text(encoding="utf-8") == payload

        assert result.iterations == 2
        assert result.text == "Wrote hello.py."
        assert result.truncated is False
        assert len(result.tool_calls) == 1
        call = result.tool_calls[0]
        assert call.name == "write_file"
        assert call.arguments == {"path": "hello.py", "content": payload}
        assert call.result is not None
        assert call.result.ok is True
        assert call.result.output.startswith("Wrote ")
        assert result.usage == Usage(input_tokens=60, output_tokens=20)
        assert client.usage == result.usage

        # Two requests were made, with the tool schemas attached.
        assert len(requests) == 2
        assert {tool["name"] for tool in requests[0]["tools"]} >= {"write_file", "read_file"}
        assert requests[0]["system"] == "you are a worker"

        # Request 2 replays the assistant turn, then feeds the tool result back
        # with the matching tool_use_id.
        second_messages = requests[1]["messages"]
        assert [message["role"] for message in second_messages] == ["user", "assistant", "user"]
        assert second_messages[1]["content"] == [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "write_file",
                "input": {"path": "hello.py", "content": payload},
            }
        ]
        tool_result = second_messages[2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "toolu_1"
        assert tool_result["is_error"] is False
        assert tool_result["content"] == call.result.output


async def test_anthropic_tool_loop_reports_a_failing_tool_without_stopping(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    cfg = anthropic_settings(make_settings)
    tools = WorkspaceTools(tmp_path)
    responses = [
        anthropic_response(
            [tool_use_block("toolu_1", "read_file", {"path": "missing.txt"})],
            stop_reason="tool_use",
        ),
        anthropic_response([text_block("That file does not exist.")]),
    ]

    async with anthropic_client(cfg, responses, tools=tools) as (client, requests):
        result = await client.complete_with_tools("read missing.txt")

    assert result.iterations == 2
    assert result.tool_calls[0].result is not None
    assert result.tool_calls[0].result.ok is False
    tool_result = requests[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "missing.txt" in tool_result["content"]


async def test_anthropic_tool_loop_bills_only_the_rounds_it_ran(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    cfg = anthropic_settings(make_settings)
    responses = [
        anthropic_response(
            [tool_use_block("toolu_1", "list_files", {})], input_tokens=3, output_tokens=4
        ),
        anthropic_response([text_block("nothing to do")], input_tokens=8, output_tokens=1),
    ]

    async with anthropic_client(cfg, responses, tools=WorkspaceTools(tmp_path)) as (client, _r):
        result = await client.complete_with_tools("look around")

    assert result.usage == Usage(input_tokens=11, output_tokens=5)


# --------------------------------------------------------------------------
# 9. the OpenAI agentic tool loop
# --------------------------------------------------------------------------


async def test_openai_tool_loop_writes_the_file_then_answers(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    cfg = openai_settings(make_settings)
    tools = WorkspaceTools(tmp_path)
    payload = "hello from openai\n"
    responses = [
        openai_response(
            text=None,
            tool_calls=[
                openai_tool_call(
                    "call_1",
                    "write_file",
                    '{"path": "note.txt", "content": "hello from openai\\n"}',
                )
            ],
            finish_reason="tool_calls",
            prompt_tokens=9,
            completion_tokens=12,
        ),
        openai_response(text="Wrote note.txt.", prompt_tokens=30, completion_tokens=4),
    ]

    async with openai_client(cfg, responses, tools=tools) as (client, requests):
        result = await client.complete_with_tools("write note.txt")

        written = tmp_path / "note.txt"
        assert written.exists()
        assert written.read_text(encoding="utf-8") == payload

        assert result.iterations == 2
        assert result.text == "Wrote note.txt."
        assert result.provider == Provider.OPENAI.value
        assert len(result.tool_calls) == 1
        call = result.tool_calls[0]
        assert call.name == "write_file"
        assert call.arguments == {"path": "note.txt", "content": payload}
        assert call.result is not None
        assert call.result.ok is True
        assert result.usage == Usage(input_tokens=39, output_tokens=16)
        assert client.usage == result.usage

        assert len(requests) == 2
        sent_tools = requests[0]["tools"]
        assert {tool["function"]["name"] for tool in sent_tools} >= {"write_file", "list_files"}
        assert "parameters" in sent_tools[0]["function"]

        second_messages = requests[1]["messages"]
        assert [message["role"] for message in second_messages] == [
            "system",
            "user",
            "assistant",
            "tool",
        ]
        assert second_messages[0]["content"] == DEFAULT_SYSTEM_PROMPT
        assert second_messages[2]["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path": "note.txt", "content": "hello from openai\\n"}',
                },
            }
        ]
        assert second_messages[3] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": call.result.output,
        }


# --------------------------------------------------------------------------
# 10. malformed tool arguments
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_arguments", "expected_fragment"),
    [
        ('{"path": "a.txt", ', "Malformed tool arguments:"),
        ("[1, 2]", "not a JSON object"),
    ],
    ids=["invalid-json", "json-but-not-an-object"],
)
async def test_openai_malformed_tool_arguments_become_a_failed_tool_result(
    make_settings: Callable[..., Settings],
    tmp_path: Path,
    raw_arguments: str,
    expected_fragment: str,
) -> None:
    """Bad JSON from the model is a tool error, not an exception."""
    cfg = openai_settings(make_settings)
    # ``make_settings`` builds its own ``tmp_path/workspace`` directory, so the
    # tools get a fresh sibling to own; only that directory may stay empty.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    responses = [
        openai_response(
            tool_calls=[openai_tool_call("call_1", "write_file", raw_arguments)],
            finish_reason="tool_calls",
        ),
        openai_response(text="I could not do that."),
    ]

    async with openai_client(cfg, responses, tools=WorkspaceTools(workspace)) as (
        client,
        requests,
    ):
        result = await client.complete_with_tools("write a file")

        assert result.iterations == 2, "the loop recovered and finished normally"
        assert result.text == "I could not do that."
        call = result.tool_calls[0]
        assert call.name == "write_file"
        assert call.arguments == {}
        assert call.result is not None
        assert call.result.ok is False
        assert call.result.error is not None
        assert expected_fragment in call.result.error
        assert list(workspace.iterdir()) == [], "nothing may be written from bad arguments"

        tool_message = requests[1]["messages"][3]
        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == "call_1"
        assert tool_message["content"] == call.result.error


# --------------------------------------------------------------------------
# 11. iteration budget
# --------------------------------------------------------------------------


async def test_exhausted_iteration_budget_returns_instead_of_raising(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    cfg = openai_settings(make_settings)
    tool_call_response = openai_response(
        tool_calls=[openai_tool_call("call_x", "list_files", "{}")],
        finish_reason="tool_calls",
    )
    responses = [tool_call_response.model_copy(deep=True) for _ in range(3)]

    async with openai_client(cfg, responses, tools=WorkspaceTools(tmp_path)) as (client, requests):
        result = await client.complete_with_tools("keep going", max_iterations=3)

        assert result.iterations == 3
        assert len(requests) == 3, "exactly one request per iteration"
        assert len(result.tool_calls) == 3
        assert all(call.result is not None and call.result.ok for call in result.tool_calls)
        assert "iteration budget of 3 was exhausted" in result.text
        assert result.truncated is False


async def test_anthropic_iteration_budget_exhaustion_returns(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    cfg = anthropic_settings(make_settings)
    responses = [
        anthropic_response(
            [tool_use_block(f"toolu_{index}", "list_files", {})],
            stop_reason="tool_use",
        )
        for index in range(2)
    ]

    async with anthropic_client(cfg, responses, tools=WorkspaceTools(tmp_path)) as (client, _r):
        result = await client.complete_with_tools("keep going", max_iterations=2)

    assert result.iterations == 2
    assert len(result.tool_calls) == 2
    assert "iteration budget of 2 was exhausted" in result.text


# --------------------------------------------------------------------------
# 12. no tools -> plain completion
# --------------------------------------------------------------------------


async def test_complete_with_tools_without_tools_degrades_to_a_plain_completion(
    make_settings: Callable[..., Settings],
) -> None:
    cfg = anthropic_settings(make_settings)
    response = anthropic_response([text_block("no tools needed")], input_tokens=6, output_tokens=2)

    async with anthropic_client(cfg, [response]) as (client, requests):
        result = await client.complete_with_tools("hi", system="be brief")

    assert result.text == "no tools needed"
    assert result.iterations == 1
    assert result.tool_calls == []
    assert result.usage == Usage(input_tokens=6, output_tokens=2)
    assert "tools" not in requests[0], "no tool schema may be sent without tools"
    assert requests[0]["messages"] == [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------
# 13. aclose()
# --------------------------------------------------------------------------


class _FakeSDKClient:
    """Stands in for ``AsyncAnthropic`` / ``AsyncOpenAI``, recording closes."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


@pytest.fixture
def fake_sdks(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[_FakeSDKClient]]:
    """Replace both SDK client classes; yields the list of constructed clients."""
    built: list[_FakeSDKClient] = []

    def _factory(**kwargs: Any) -> _FakeSDKClient:
        client = _FakeSDKClient(**kwargs)
        built.append(client)
        return client

    monkeypatch.setattr(llm_module, "AsyncAnthropic", _factory)
    monkeypatch.setattr(llm_module, "AsyncOpenAI", _factory)
    return built


async def test_aclose_is_safe_to_call_twice(
    make_settings: Callable[..., Settings], fake_sdks: list[_FakeSDKClient]
) -> None:
    cfg = make_settings(provider="anthropic", anthropic_api_key=DUMMY_KEY, openai_api_key="k")
    client = LLMClient(cfg)

    assert fake_sdks == [], "nothing is built until a client is asked for"
    _ = client.anthropic
    _ = client.openai
    assert len(fake_sdks) == 2

    await client.aclose()
    await client.aclose()

    assert [sdk.close_calls for sdk in fake_sdks] == [1, 1], "closed once, not twice"
    assert client._anthropic is None
    assert client._openai is None


async def test_aclose_survives_a_failing_sdk_close(
    make_settings: Callable[..., Settings], fake_sdks: list[_FakeSDKClient]
) -> None:
    """Teardown is best-effort: one broken client must not block the others."""
    cfg = make_settings(provider="anthropic", anthropic_api_key=DUMMY_KEY, openai_api_key="k")
    client = LLMClient(cfg)
    _ = client.anthropic
    _ = client.openai

    async def _explode() -> None:
        raise RuntimeError("close failed")

    fake_sdks[0].close = _explode  # type: ignore[method-assign]

    await client.aclose()

    assert fake_sdks[1].close_calls == 1
