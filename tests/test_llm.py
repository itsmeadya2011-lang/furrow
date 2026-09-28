from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from furrow.config import Provider, Settings
from furrow.llm import (
    CompletionResult,
    LLMClient,
    _parse_anthropic_response,
    _parse_openai_response,
    _to_anthropic_messages,
    _to_anthropic_tools,
    _to_openai_messages,
    _to_openai_tools,
)
from furrow.tools import TOOL_SCHEMAS


def _tc(id: str, name: str, args: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(id=id, name=name, input=args, type="tool_use", text=None)


def _openai_tc(id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_to_anthropic_messages_with_tool_calls_and_results():
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "do something"},
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [{"id": "tc1", "name": "read_file", "args": {"path": "x"}}],
        },
        {"role": "tool", "content": "file contents", "tool_call_id": "tc1"},
    ]
    native, system = _to_anthropic_messages(messages)
    assert system == "you are helpful"
    assert native[0] == {"role": "user", "content": "do something"}
    asm = native[1]
    assert asm["role"] == "assistant"
    content = asm["content"]
    assert content[0] == {"type": "text", "text": "ok"}
    assert content[1] == {
        "type": "tool_use",
        "id": "tc1",
        "name": "read_file",
        "input": {"path": "x"},
    }
    assert native[2]["role"] == "user"
    assert native[2]["content"][0]["type"] == "tool_result"
    assert native[2]["content"][0]["tool_use_id"] == "tc1"
    assert native[2]["content"][0]["content"] == "file contents"


def test_to_anthropic_messages_no_system():
    native, system = _to_anthropic_messages([{"role": "user", "content": "hi"}])
    assert system == ""
    assert native == [{"role": "user", "content": "hi"}]


def test_to_openai_messages_with_system_and_tools():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "tc1", "name": "read_file", "args": {"path": "x"}}],
        },
        {"role": "tool", "content": "res", "tool_call_id": "tc1"},
    ]
    native = _to_openai_messages(messages, system="sys")
    assert native[0] == {"role": "system", "content": "sys"}
    assert native[1] == {"role": "user", "content": "hi"}
    asm = native[2]
    assert asm["role"] == "assistant"
    assert asm["content"] == ""
    assert asm["tool_calls"][0] == {
        "id": "tc1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path": "x"}'},
    }
    assert native[3] == {"role": "tool", "content": "res", "tool_call_id": "tc1"}


def test_to_anthropic_tools_shape():
    tools = _to_anthropic_tools(TOOL_SCHEMAS)
    names = {t["name"] for t in tools}
    assert names == {"read_file", "write_file", "edit_file", "list_files", "run_command"}
    first = next(t for t in tools if t["name"] == "read_file")
    assert first["type"] == "tool"
    assert first["input_schema"]["type"] == "object"
    assert "path" in first["input_schema"]["properties"]


def test_to_openai_tools_shape():
    tools = _to_openai_tools(TOOL_SCHEMAS)
    first = next(t for t in tools if t["function"]["name"] == "run_command")
    assert first["type"] == "function"
    assert first["function"]["parameters"]["required"] == ["command"]
    names = {t["function"]["name"] for t in tools}
    assert names == {"read_file", "write_file", "edit_file", "list_files", "run_command"}


def test_parse_anthropic_text():
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello world", id=None, name=None, input=None)]
    )
    result = _parse_anthropic_response(resp)
    assert isinstance(result, CompletionResult)
    assert result.text == "hello world"
    assert result.tool_calls == []


def test_parse_anthropic_tool_calls():
    resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="let me check", id=None, name=None, input=None),
            SimpleNamespace(
                type="tool_use", id="uc1", name="read_file", input={"path": "x"}, text=None
            ),
        ]
    )
    result = _parse_anthropic_response(resp)
    assert result.text == "let me check"
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.id == "uc1"
    assert tc.name == "read_file"
    assert tc.args == {"path": "x"}


def test_parse_openai_text():
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi", tool_calls=[]))]
    )
    result = _parse_openai_response(resp)
    assert result.text == "hi"
    assert result.tool_calls == []


def test_parse_openai_tool_calls():
    tc = _openai_tc("oc1", "write_file", '{"path": "a.py", "content": "print(1)"}')
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[tc]))]
    )
    result = _parse_openai_response(resp)
    assert result.text == "ok"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "oc1"
    assert result.tool_calls[0].name == "write_file"
    assert result.tool_calls[0].args == {"path": "a.py", "content": "print(1)"}


def test_parse_openai_malformed_arguments():
    tc = _openai_tc("oc1", "write_file", "not valid json")
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[tc]))]
    )
    result = _parse_openai_response(resp)
    assert result.tool_calls[0].args == {}


@pytest.mark.asyncio
async def test_complete_with_tools_ollama_unsupported():
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="not supported"):
        await client.complete_with_tools(
            [{"role": "user", "content": "hi"}],
            TOOL_SCHEMAS,
            system="",
        )
