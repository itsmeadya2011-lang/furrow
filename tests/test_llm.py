from __future__ import annotations

import asyncio
from typing import Any

import anthropic
import httpx
import pytest

from furrow.config import Provider, Settings
from furrow.llm import (
    LLMClient,
    ToolCall,
    _is_retryable,
    parse_tool_arguments,
    to_anthropic_messages,
    to_anthropic_tools,
    to_openai_messages,
    to_openai_tools,
)


def test_settings_none_uses_global_settings():
    from furrow.config import settings as global_settings

    client = LLMClient(settings=None)
    assert client.settings is global_settings
    assert client.settings.planner_model == "claude-3-5-haiku-20241022"


def test_settings_instance_is_stored():
    custom = Settings(provider=Provider.OPENAI, model="gpt-4o-mini")
    client = LLMClient(settings=custom)
    assert client.settings is custom
    assert client.settings.provider == Provider.OPENAI
    assert client.settings.model == "gpt-4o-mini"


def test_tool_call_dataclass():
    tc = ToolCall(id="t1", name="read_file", arguments={"path": "x.py"})
    assert tc.id == "t1"
    assert tc.name == "read_file"
    assert tc.arguments == {"path": "x.py"}


def test_to_anthropic_tools():
    tools = [
        {
            "name": "grep",
            "description": "Search files",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    assert to_anthropic_tools(tools) == [
        {
            "name": "grep",
            "description": "Search files",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


def test_to_openai_tools():
    tools = [
        {
            "name": "grep",
            "description": "Search files",
            "parameters": {"type": "object"},
        }
    ]
    assert to_openai_tools(tools) == [
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search files",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_to_anthropic_messages_plain_user_text():
    messages = [{"role": "user", "content": "hello world"}]
    assert to_anthropic_messages(messages) == [{"role": "user", "content": "hello world"}]


def test_to_anthropic_messages_assistant_text_and_tool_use():
    messages = [
        {"role": "user", "content": "edit the file"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will edit the file."},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "write_file",
                    "input": {"path": "a.txt", "content": "hi"},
                },
            ],
        },
    ]
    assert to_anthropic_messages(messages) == [
        {"role": "user", "content": "edit the file"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will edit the file."},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "write_file",
                    "input": {"path": "a.txt", "content": "hi"},
                },
            ],
        },
    ]


def test_to_anthropic_messages_user_tool_result():
    messages = [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}
    ]
    assert to_anthropic_messages(messages) == [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
        }
    ]


def test_to_openai_messages_plain_user_text():
    messages = [{"role": "user", "content": "hello world"}]
    assert to_openai_messages(messages) == [{"role": "user", "content": "hello world"}]


def test_to_openai_messages_assistant_text_and_tool_use():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will edit the file."},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "write_file",
                    "input": {"path": "a.txt", "content": "hi"},
                },
            ],
        }
    ]
    # When both text and tool_calls are present, they are emitted as two
    # separate assistant messages to preserve the order in which the blocks
    # appeared in the original content.
    assert to_openai_messages(messages) == [
        {"role": "assistant", "content": "I will edit the file."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        'arguments': '{"path": "a.txt", "content": "hi"}',
                    },
                }
            ],
        },
    ]


def test_to_openai_messages_user_tool_result():
    messages = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "tool output"}],
        },
    ]
    assert to_openai_messages(messages) == [
        {"role": "tool", "tool_call_id": "t1", "content": "tool output"},
    ]


def test_to_openai_messages_user_tool_result_with_text():
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "here is the result"},
            {"type": "tool_result", "tool_use_id": "t1", "content": "tool output"},
        ]}
    ]
    assert to_openai_messages(messages) == [
        {"role": "tool", "tool_call_id": "t1", "content": "tool output"},
        {"role": "user", "content": "here is the result"},
    ]


def test_to_openai_messages_assistant_tool_only():
    """An assistant message with only tool_use blocks is emitted as a single message."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "write_file",
                    "input": {"path": "a.txt", "content": "hi"},
                },
            ],
        }
    ]
    assert to_openai_messages(messages) == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        'arguments': '{"path": "a.txt", "content": "hi"}',
                    },
                }
            ],
        }
    ]


def test_to_openai_messages_assistant_text_only():
    """An assistant message with only text blocks is emitted as a single message."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "thinking..."},
                {"type": "text", "text": "more text"},
            ],
        }
    ]
    assert to_openai_messages(messages) == [
        {"role": "assistant", "content": "thinking...\nmore text"},
    ]


def test_to_openai_messages_preserves_block_ordering():
    """When text and tool_use blocks are interleaved, order is preserved.

    The original implementation collapsed text and tool_calls into a single
    assistant message, which lost the ordering information. The new
    implementation emits them as separate assistant messages to preserve the
    sequence in which the blocks appeared.
    """
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "first"},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "read_file",
                    "input": {"path": "a.txt"},
                },
                {"type": "text", "text": "second"},
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "write_file",
                    "input": {"path": "b.txt", "content": "x"},
                },
            ],
        }
    ]
    result = to_openai_messages(messages)
    # The blocks are emitted in order: text, tool_call, text, tool_call.
    assert len(result) == 4
    assert result[0] == {"role": "assistant", "content": "first"}
    assert result[1]["role"] == "assistant"
    assert result[1]["tool_calls"][0]["id"] == "t1"
    assert result[2] == {"role": "assistant", "content": "second"}
    assert result[3]["tool_calls"][0]["id"] == "t2"


def test_parse_tool_arguments_dict():
    assert parse_tool_arguments({"a": 1}) == {"a": 1}


def test_parse_tool_arguments_json_string():
    assert parse_tool_arguments('{"a": 1}') == {"a": 1}


def test_parse_tool_arguments_invalid_json():
    assert parse_tool_arguments("not json") == {}
    assert parse_tool_arguments("[1, 2]") == {}
    assert parse_tool_arguments(42) == {}
    assert parse_tool_arguments(None) == {}


class _FakeStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeAnthropicTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeAnthropicUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeAnthropicResponse:
    def __init__(self, text: str, usage: _FakeAnthropicUsage | None = None) -> None:
        self.content = [_FakeAnthropicTextBlock(text)]
        self.usage = usage
        self.stop_reason = "end_turn"


class _FakeAnthropicMessages:
    def __init__(self, response: _FakeAnthropicResponse) -> None:
        self._response = response

    async def create(self, **kwargs: object) -> _FakeAnthropicResponse:
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response: _FakeAnthropicResponse) -> None:
        self.messages = _FakeAnthropicMessages(response)


class _FakeOpenAIUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls: list[Any] | None = None


class _FakeChoice:
    def __init__(self, content: str, usage: _FakeOpenAIUsage | None = None) -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = "stop"
        self.usage = usage


class _FakeResponse:
    def __init__(self, content: str, usage: _FakeOpenAIUsage | None = None) -> None:
        self.choices = [_FakeChoice(content, usage)]
        self.usage = usage


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.last_kwargs: dict = {}
        self.usage: _FakeOpenAIUsage | None = None

    async def create(self, **kwargs) -> _FakeResponse:
        self.last_kwargs = kwargs
        return _FakeResponse(self.content, self.usage)


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)


class _FakeOpenAI:
    def __init__(self, content: str = "ok") -> None:
        self.chat = _FakeChat(content)


def test_is_retryable_timeout_and_connection_error():
    assert _is_retryable(asyncio.TimeoutError()) is True
    assert _is_retryable(ConnectionError()) is True


def test_is_retryable_status_codes():
    assert _is_retryable(_FakeStatusError(429)) is True
    assert _is_retryable(_FakeStatusError(409)) is True
    assert _is_retryable(_FakeStatusError(503)) is True
    assert _is_retryable(_FakeStatusError(400)) is False
    assert _is_retryable(_FakeStatusError(401)) is False
    assert _is_retryable(_FakeStatusError(404)) is False
    assert _is_retryable(ValueError("not retryable")) is False


def test_is_retryable_real_anthropic_api_status_error():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    try:
        response = httpx.Response(429, request=request)
        exc = anthropic.APIStatusError("rate limited", response=response, body={})
    except Exception:  # pragma: no cover - sdk shape dependent
        pytest.skip("cannot construct anthropic.APIStatusError in this sdk version")
    assert _is_retryable(exc) is True

    response_400 = httpx.Response(400, request=request)
    exc_400 = anthropic.APIStatusError("bad request", response=response_400, body={})
    assert _is_retryable(exc_400) is False


async def test_complete_ollama_does_not_raise_unsupported_provider():
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
    fake = _FakeOpenAI("ok")
    client._ollama = fake

    result = await client.complete("please edit a file")

    assert result == "ok"
    assert fake.chat.completions.last_kwargs["model"] is not None


def test_ollama_base_url_ends_with_v1():
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
    assert str(client.ollama.base_url).rstrip("/") == "http://localhost:11434/v1"

    client_custom = LLMClient(
        settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434/")
    )
    assert str(client_custom.ollama.base_url).rstrip("/") == "http://localhost:11434/v1"


def test_usage_total_tokens():
    from furrow.config import Usage

    usage = Usage(input_tokens=10, output_tokens=5)
    assert usage.total_tokens == 15
    assert usage.calls == 0


def test_usage_add_accumulates_and_returns_self():
    from furrow.config import Usage

    a = Usage(input_tokens=10, output_tokens=5, calls=2)
    b = Usage(input_tokens=3, output_tokens=7, calls=1)
    result = a.add(b)
    assert result is a
    assert a.input_tokens == 13
    assert a.output_tokens == 12
    assert a.calls == 3
    assert a.total_tokens == 25


def test_usage_defaults_zero():
    from furrow.config import Usage

    u = Usage()
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.calls == 0
    assert u.total_tokens == 0


async def test_complete_anthropic_extracts_usage():
    usage = _FakeAnthropicUsage(input_tokens=100, output_tokens=42)
    response = _FakeAnthropicResponse("hello there", usage=usage)
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))
    client._anthropic = _FakeAnthropicClient(response)

    result = await client.complete("hi")

    assert result == "hello there"
    assert client.total_usage.input_tokens == 100
    assert client.total_usage.output_tokens == 42
    assert client.total_usage.calls == 1
    assert client.total_usage.total_tokens == 142


async def test_complete_openai_extracts_usage():
    from furrow.config import Provider

    fake_usage = _FakeOpenAIUsage(prompt_tokens=77, completion_tokens=33)
    fake = _FakeOpenAI("ok")
    fake.chat.completions.usage = fake_usage
    client = LLMClient(settings=Settings(provider=Provider.OPENAI))
    client._openai = fake

    result = await client.complete("hi")

    assert result == "ok"
    assert client.total_usage.input_tokens == 77
    assert client.total_usage.output_tokens == 33
    assert client.total_usage.calls == 1
    assert client.total_usage.total_tokens == 110


async def test_complete_without_usage_does_not_crash():
    from furrow.config import Provider

    fake = _FakeOpenAI("ok")
    client = LLMClient(settings=Settings(provider=Provider.OPENAI))
    client._openai = fake

    result = await client.complete("hi")

    assert result == "ok"
    assert client.total_usage.input_tokens == 0
    assert client.total_usage.output_tokens == 0
    assert client.total_usage.calls == 1
    assert client.total_usage.total_tokens == 0


async def test_total_usage_accumulates_across_calls():
    from furrow.config import Provider

    fake = _FakeOpenAI("ok")
    fake.chat.completions.usage = _FakeOpenAIUsage(prompt_tokens=10, completion_tokens=5)
    client = LLMClient(settings=Settings(provider=Provider.OPENAI))
    client._openai = fake

    await client.complete("first")
    await client.complete("second")

    assert client.total_usage.calls == 2
    assert client.total_usage.input_tokens == 20
    assert client.total_usage.output_tokens == 10
    assert client.total_usage.total_tokens == 30


async def test_aclose_closes_created_clients():
    client = LLMClient(settings=Settings(provider=Provider.OPENAI))
    fake = _FakeOpenAI("ok")

    closed = {"n": 0}

    async def aclose() -> None:
        closed["n"] += 1

    fake.aclose = aclose
    client._openai = fake

    await client.aclose()

    assert closed["n"] == 1
    assert client._openai is None


async def test_aclose_no_client_is_noop():
    client = LLMClient(settings=Settings(provider=Provider.OPENAI))
    await client.aclose()  # should not raise
    assert client._openai is None


async def test_aclose_twice_does_not_raise():
    client = LLMClient(settings=Settings(provider=Provider.OPENAI))
    fake = _FakeOpenAI("ok")

    async def aclose() -> None:
        return None

    fake.aclose = aclose
    client._openai = fake

    await client.aclose()
    await client.aclose()  # second call must not raise


async def test_aclose_swallows_close_errors():
    client = LLMClient(settings=Settings(provider=Provider.OPENAI))
    fake = _FakeOpenAI("ok")

    async def aclose() -> None:
        raise RuntimeError("boom")

    fake.aclose = aclose
    client._openai = fake

    await client.aclose()  # must not raise
    assert client._openai is None


async def test_aclose_resets_so_fresh_client_created():
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
    fake = _FakeOpenAI("ok")

    async def aclose() -> None:
        return None

    fake.aclose = aclose
    client._ollama = fake

    await client.aclose()
    assert client._ollama is None

    fresh = client.ollama
    assert fresh is not fake

class _FakeResponseNoChoices:
    def __init__(self) -> None:
        self.choices = []
        self.usage = None


class _FakeChoiceNoMessage:
    def __init__(self) -> None:
        self.message = None
        self.finish_reason = "stop"


class _FakeResponseNoMessage:
    def __init__(self) -> None:
        self.choices = [_FakeChoiceNoMessage()]
        self.usage = None


def test_is_retryable_response_status_code():
    class _RespErr(Exception):
        def __init__(self, code: int) -> None:
            self.response = type("R", (), {"status_code": code})()

    assert _is_retryable(_RespErr(429)) is True
    assert _is_retryable(_RespErr(503)) is True
    assert _is_retryable(_RespErr(400)) is False


def test_is_retryable_no_status_at_all():
    assert _is_retryable(ValueError("no status")) is False


def test_is_retryable_status_code_path_still_works():
    assert _is_retryable(_FakeStatusError(429)) is True
    assert _is_retryable(_FakeStatusError(400)) is False


async def test_complete_openai_empty_choices_raises_valueerror():
    from furrow.config import Provider

    fake = _FakeOpenAI("ok")
    fake.chat.completions = type("C", (), {})()
    async def create(**kwargs):
        return _FakeResponseNoChoices()
    fake.chat.completions.create = create
    client = LLMClient(settings=Settings(provider=Provider.OPENAI))
    client._openai = fake
    with pytest.raises(ValueError, match="no choices"):
        await client.complete("hi")


async def test_chat_openai_empty_choices_raises_valueerror():
    from furrow.config import Provider

    fake = _FakeOpenAI("ok")
    fake.chat.completions = type("C", (), {})()
    async def create(**kwargs):
        return _FakeResponseNoChoices()
    fake.chat.completions.create = create
    client = LLMClient(settings=Settings(provider=Provider.OPENAI))
    client._openai = fake
    with pytest.raises(ValueError, match="no choices"):
        await client.chat([{"role": "user", "content": "hi"}])


async def test_complete_openai_choice_no_message_raises_valueerror():
    from furrow.config import Provider

    fake = _FakeOpenAI("ok")
    fake.chat.completions = type("C", (), {})()
    async def create(**kwargs):
        return _FakeResponseNoMessage()
    fake.chat.completions.create = create
    client = LLMClient(settings=Settings(provider=Provider.OPENAI))
    client._openai = fake
    with pytest.raises(ValueError, match="no message"):
        await client.complete("hi")


async def test_chat_openai_choice_no_message_raises_valueerror():
    from furrow.config import Provider

    fake = _FakeOpenAI("ok")
    fake.chat.completions = type("C", (), {})()
    async def create(**kwargs):
        return _FakeResponseNoMessage()
    fake.chat.completions.create = create
    client = LLMClient(settings=Settings(provider=Provider.OPENAI))
    client._openai = fake
    with pytest.raises(ValueError, match="no message"):
        await client.chat([{"role": "user", "content": "hi"}])


async def test_complete_anthropic_none_content_yields_empty():
    class _Resp:
        content = None
        usage = None
        stop_reason = "end_turn"

    class _Msgs:
        async def create(self, **kwargs):
            return _Resp()

    class _Client:
        messages = _Msgs()

    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))
    client._anthropic = _Client()
    result = await client.complete("hi")
    assert result == ""


async def test_chat_anthropic_none_content_yields_empty():
    class _Resp:
        content = None
        usage = None
        stop_reason = "end_turn"

    class _Msgs:
        async def create(self, **kwargs):
            return _Resp()

    class _Client:
        messages = _Msgs()

    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))
    client._anthropic = _Client()
    result = await client.chat([{"role": "user", "content": "hi"}])
    assert result.text == ""


@pytest.fixture
def fast_sleep(monkeypatch):
    """Replace ``asyncio.sleep`` with a no-op so retry back-off does not slow tests."""
    delays: list[float] = []

    async def fake_sleep(seconds):
        delays.append(float(seconds))

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return delays


@pytest.mark.parametrize("tenacity_available", [True, False])
async def test_retry_retries_a_retryable_error_the_expected_number_of_times(
    monkeypatch, fast_sleep, tenacity_available
):
    monkeypatch.setattr("furrow.llm._TENACITY_AVAILABLE", tenacity_available)
    calls = []

    async def call() -> str:
        calls.append(1)
        raise _FakeStatusError(503)

    client = LLMClient(settings=Settings(provider=Provider.OPENAI, max_retries=3))
    with pytest.raises(_FakeStatusError):
        await client._retry(call)
    assert len(calls) == 3  # max_retries attempts


@pytest.mark.parametrize("tenacity_available", [True, False])
async def test_retry_raises_a_non_retryable_error_immediately(
    monkeypatch, fast_sleep, tenacity_available
):
    monkeypatch.setattr("furrow.llm._TENACITY_AVAILABLE", tenacity_available)
    calls = []

    async def call() -> str:
        calls.append(1)
        raise ValueError("not retryable")

    client = LLMClient(settings=Settings(provider=Provider.OPENAI, max_retries=3))
    with pytest.raises(ValueError, match="not retryable"):
        await client._retry(call)
    assert len(calls) == 1  # a non-retryable error is never retried


@pytest.mark.parametrize("tenacity_available", [True, False])
async def test_retry_returns_the_result_of_a_successful_call(
    monkeypatch, fast_sleep, tenacity_available
):
    monkeypatch.setattr("furrow.llm._TENACITY_AVAILABLE", tenacity_available)
    calls = []

    async def call() -> str:
        calls.append(1)
        return "ok"

    client = LLMClient(settings=Settings(provider=Provider.OPENAI, max_retries=3))
    assert await client._retry(call) == "ok"
    assert len(calls) == 1


@pytest.mark.parametrize("tenacity_available", [True, False])
async def test_retry_succeeds_after_a_transient_error(
    monkeypatch, fast_sleep, tenacity_available
):
    monkeypatch.setattr("furrow.llm._TENACITY_AVAILABLE", tenacity_available)
    calls = []

    async def call() -> str:
        calls.append(1)
        if len(calls) < 2:
            raise _FakeStatusError(503)
        return "recovered"

    client = LLMClient(settings=Settings(provider=Provider.OPENAI, max_retries=3))
    assert await client._retry(call) == "recovered"
    assert len(calls) == 2


async def test_complete_retries_a_transient_provider_error(monkeypatch, fast_sleep):
    class _FlakyCompletions:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise _FakeStatusError(503)
            return _FakeResponse("ok")

    class _FlakyChat:
        def __init__(self) -> None:
            self.completions = _FlakyCompletions()

    class _FlakyOpenAI:
        def __init__(self) -> None:
            self.chat = _FlakyChat()

    fake = _FlakyOpenAI()
    client = LLMClient(settings=Settings(provider=Provider.OPENAI, max_retries=3))
    client._openai = fake

    result = await client.complete("hi")

    assert result == "ok"
    assert fake.chat.completions.calls == 2


async def test_complete_anthropic_records_the_cost_of_the_call():
    usage = _FakeAnthropicUsage(input_tokens=100, output_tokens=42)
    response = _FakeAnthropicResponse("hello there", usage=usage)
    client = LLMClient(
        settings=Settings(provider=Provider.ANTHROPIC, model="claude-3-5-haiku-20241022")
    )
    client._anthropic = _FakeAnthropicClient(response)

    result = await client.complete("hi")

    assert result == "hello there"
    # claude-3-5-haiku: $0.80 / $4.00 per 1,000,000 tokens
    expected = 100 / 1e6 * 0.80 + 42 / 1e6 * 4.0
    assert client.total_usage.cost_usd == pytest.approx(expected)
    assert client.total_usage.cost_usd > 0


async def test_complete_openai_records_the_cost_of_the_call():
    fake = _FakeOpenAI("ok")
    fake.chat.completions.usage = _FakeOpenAIUsage(prompt_tokens=77, completion_tokens=33)
    client = LLMClient(settings=Settings(provider=Provider.OPENAI, model="gpt-4o-mini"))
    client._openai = fake

    result = await client.complete("hi")

    assert result == "ok"
    # gpt-4o-mini: $0.15 / $0.60 per 1,000,000 tokens
    expected = round(77 / 1e6 * 0.15 + 33 / 1e6 * 0.60, 6)
    assert client.total_usage.cost_usd == pytest.approx(expected)
    assert client.total_usage.cost_usd > 0


async def test_cost_uses_the_model_that_was_actually_requested():
    fake = _FakeOpenAI("ok")
    fake.chat.completions.usage = _FakeOpenAIUsage(prompt_tokens=1000, completion_tokens=1000)
    client = LLMClient(settings=Settings(provider=Provider.OPENAI, model="gpt-4o"))
    client._openai = fake

    await client.complete("hi", model="gpt-4o-mini")

    # priced as gpt-4o-mini, not as the configured gpt-4o
    expected = 1000 / 1e6 * 0.15 + 1000 / 1e6 * 0.60
    assert client.total_usage.cost_usd == pytest.approx(expected)


async def test_unknown_model_leaves_the_cost_at_zero_but_the_call_succeeds():
    fake = _FakeOpenAI("ok")
    fake.chat.completions.usage = _FakeOpenAIUsage(prompt_tokens=1000, completion_tokens=500)
    client = LLMClient(
        settings=Settings(provider=Provider.OPENAI, model="totally-unknown-model")
    )
    client._openai = fake

    result = await client.complete("hi")

    assert result == "ok"
    assert client.total_usage.calls == 1
    assert client.total_usage.cost_usd == 0.0


async def test_chat_anthropic_sets_the_cost_on_the_per_call_usage():
    usage = _FakeAnthropicUsage(input_tokens=100, output_tokens=42)
    response = _FakeAnthropicResponse("hello there", usage=usage)
    client = LLMClient(
        settings=Settings(provider=Provider.ANTHROPIC, model="claude-3-5-haiku-20241022")
    )
    client._anthropic = _FakeAnthropicClient(response)

    result = await client.chat([{"role": "user", "content": "hi"}])

    expected = 100 / 1e6 * 0.80 + 42 / 1e6 * 4.0
    assert result.usage.cost_usd == pytest.approx(expected)
    assert client.total_usage.cost_usd == pytest.approx(result.usage.cost_usd)


async def test_chat_openai_sets_the_cost_on_the_per_call_usage():
    fake = _FakeOpenAI("ok")
    fake.chat.completions.usage = _FakeOpenAIUsage(prompt_tokens=77, completion_tokens=33)
    client = LLMClient(settings=Settings(provider=Provider.OPENAI, model="gpt-4o-mini"))
    client._openai = fake

    result = await client.chat([{"role": "user", "content": "hi"}])

    expected = round(77 / 1e6 * 0.15 + 33 / 1e6 * 0.60, 6)
    assert result.usage.cost_usd == pytest.approx(expected)
    assert client.total_usage.cost_usd == pytest.approx(result.usage.cost_usd)


async def test_a_broken_price_table_does_not_break_the_request(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("pricing exploded")

    monkeypatch.setattr("furrow.llm.estimate_cost", _boom)
    fake = _FakeOpenAI("ok")
    fake.chat.completions.usage = _FakeOpenAIUsage(prompt_tokens=10, completion_tokens=5)
    client = LLMClient(settings=Settings(provider=Provider.OPENAI, model="gpt-4o-mini"))
    client._openai = fake

    result = await client.complete("hi")

    assert result == "ok"
    assert client.total_usage.cost_usd == 0.0


async def test_pricing_overrides_are_used_for_the_cost():
    fake = _FakeOpenAI("ok")
    fake.chat.completions.usage = _FakeOpenAIUsage(prompt_tokens=1000, completion_tokens=1000)
    client = LLMClient(
        settings=Settings(
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
            model_pricing={
                "gpt-4o-mini": {"input_per_million": 1.0, "output_per_million": 2.0}
            },
        )
    )
    client._openai = fake

    await client.complete("hi")

    expected = 1000 / 1e6 * 1.0 + 1000 / 1e6 * 2.0
    assert client.total_usage.cost_usd == pytest.approx(expected)
