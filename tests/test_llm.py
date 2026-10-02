"""Tests for the Furrow LLM client."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient
from furrow.tools.base import Tool, ToolResult


class DummyTool(Tool):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "A dummy tool"

    @property
    def _properties(self) -> dict:
        return {"value": {"type": "string"}}

    @property
    def _required(self) -> list[str]:
        return ["value"]

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=f"dummy got: {kwargs.get('value')}")


# --- _execute_tool ---


@pytest.mark.asyncio
async def test_execute_tool_success():
    client = LLMClient(settings=Settings())
    call = {"id": "1", "name": "dummy", "input": {"value": "hello"}}
    result = await client._execute_tool(call, [DummyTool()])
    assert "dummy got: hello" in result


@pytest.mark.asyncio
async def test_execute_tool_not_found():
    client = LLMClient(settings=Settings())
    call = {"id": "1", "name": "nonexistent", "input": {}}
    result = await client._execute_tool(call, [DummyTool()])
    assert "Tool not found" in result


@pytest.mark.asyncio
async def test_execute_tool_failure():
    client = LLMClient(settings=Settings())

    class FailingTool(Tool):
        @property
        def name(self) -> str:
            return "fail"

        @property
        def description(self) -> str:
            return "A failing tool"

        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult(success=False, output="", error="something went wrong")

    call = {"id": "1", "name": "fail", "input": {}}
    result = await client._execute_tool(call, [FailingTool()])
    assert "Error: something went wrong" in result


# --- complete() dispatch ---


@pytest.mark.asyncio
async def test_complete_unsupported_provider():
    settings = Settings()
    client = LLMClient(settings=settings)
    # Simulate an unsupported provider value by patching the attribute
    settings.provider = "unsupported"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hello")


# --- Provider property initialization ---


def test_anthropic_not_configured():
    settings = Settings()
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        _ = client.anthropic


def test_openai_not_configured():
    settings = Settings()
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _ = client.openai


# --- File utilities ---


@pytest.mark.asyncio
async def test_read_file_util(tmp_path: Path):
    p = tmp_path / "test.txt"
    p.write_text("content here")
    client = LLMClient(settings=Settings())
    content = await client.read_file(p)
    assert content == "content here"


@pytest.mark.asyncio
async def test_write_file_util(tmp_path: Path):
    p = tmp_path / "subdir" / "new.txt"
    client = LLMClient(settings=Settings())
    await client.write_file(p, "written content")
    assert p.read_text() == "written content"


# --- complete_with_tools dispatch ---


@pytest.mark.asyncio
async def test_complete_with_tools_delegates_to_provider():
    """Verify complete_with_tools delegates to the right provider loop."""
    settings = Settings()
    client = LLMClient(settings=settings)
    with patch.object(client, "_anthropic_tool_loop", new=AsyncMock(return_value="answer")):
        result = await client.complete_with_tools(
            prompt="Hi", system="", model="test", tools=[], max_rounds=5
        )
        assert result == "answer"


@pytest.mark.asyncio
async def test_complete_with_tools_openai_loop():
    settings = Settings()
    settings.provider = Provider.OPENAI
    client = LLMClient(settings=settings)
    with patch.object(client, "_openai_tool_loop", new=AsyncMock(return_value="openai answer")):
        result = await client.complete_with_tools(
            prompt="Hi", system="", model="test", tools=[], max_rounds=5
        )
        assert result == "openai answer"


@pytest.mark.asyncio
async def test_complete_with_tools_ollama_loop():
    settings = Settings()
    settings.provider = Provider.OLLAMA
    client = LLMClient(settings=settings)
    with patch.object(client, "_ollama_tool_loop", new=AsyncMock(return_value="ollama answer")):
        result = await client.complete_with_tools(
            prompt="Hi", system="", model="test", tools=[], max_rounds=5
        )
        assert result == "ollama answer"
