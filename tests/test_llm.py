from __future__ import annotations

import asyncio

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
    assert to_openai_messages(messages) == [
        {
            "role": "assistant",
            "content": "I will edit the file.",
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


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = "stop"


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.last_kwargs: dict = {}

    async def create(self, **kwargs) -> _FakeResponse:
        self.last_kwargs = kwargs
        return _FakeResponse(self.content)


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)


class _FakeOpenAI:
    def __init__(self, content: str = "ok") -> None:
        self.chat = _FakeChat(content)


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