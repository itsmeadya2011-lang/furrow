"""Tests for the LLM client tool-use and provider support."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from furrow.config import Provider, Settings
from furrow.llm import LLMClient, LLMResult, ToolCall


class TestSettings:
    def test_default_settings(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0
        assert s.agent_max_iterations == 10

    def test_max_cycles_override(self):
        s = Settings(max_cycles=5)
        assert s.max_cycles == 5


class TestLLMClient:
    def test_init_default(self):
        client = LLMClient()
        assert client.settings is not None
        assert client._anthropic is None
        assert client._openai is None

    def test_init_with_custom_settings(self):
        s = Settings(model="gpt-4", provider=Provider.OPENAI)
        client = LLMClient(settings=s)
        assert client.settings.model == "gpt-4"

    def test_anthropic_property_raises_without_key(self):
        s = Settings(provider=Provider.ANTHROPIC)
        client = LLMClient(settings=s)
        import os
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                _ = client.anthropic
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_ollama_client_uses_ollama_base_url(self):
        s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
        client = LLMClient(settings=s)
        oc = client.ollama
        assert oc is not None
        assert "11434" in str(oc.base_url)

    @pytest.mark.asyncio
    async def test_complete_with_tools_anthropic_no_tools(self):
        """Complete with no tools returns text only."""
        s = Settings(provider=Provider.ANTHROPIC, model="claude-3-5-haiku-20241022")
        client = LLMClient(settings=s)

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="text", text="Hello from AI")
        ]

        client._anthropic = MagicMock()
        client._anthropic.messages.create = AsyncMock(return_value=mock_response)

        result = await client.complete_with_tools(
            prompt="Hi", system="test", model="claude-3-5-haiku-20241022"
        )

        assert isinstance(result, LLMResult)
        assert result.text == "Hello from AI"
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_complete_with_tools_anthropic_with_tool_calls(self):
        s = Settings(provider=Provider.ANTHROPIC, model="claude-3-5-sonnet-20241022")
        client = LLMClient(settings=s)

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="text", text="I need to read a file."),
            MagicMock(type="tool_use", id="call_1", name="read_file", input={"path": "/tmp/test"}),
        ]

        client._anthropic = MagicMock()
        client._anthropic.messages.create = AsyncMock(return_value=mock_response)

        result = await client.complete_with_tools(
            prompt="Read a file",
            system="test",
            model="claude-3-5-sonnet-20241022",
            tools=[{"name": "read_file", "description": "Read a file", "input_schema": {}}],
        )

        assert isinstance(result, LLMResult)
        assert result.text == "I need to read a file."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].input == {"path": "/tmp/test"}

    @pytest.mark.asyncio
    async def test_complete_openai_provider(self):
        s = Settings(provider=Provider.OPENAI, model="gpt-4")
        client = LLMClient(settings=s)

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="OpenAI response", tool_calls=None)
            )
        ]

        client._openai = MagicMock()
        client._openai.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.complete_with_tools(prompt="Hi", system="", model="gpt-4")

        assert isinstance(result, LLMResult)
        assert result.text == "OpenAI response"
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_complete_openai_with_tool_calls(self):
        s = Settings(provider=Provider.OPENAI, model="gpt-4")
        client = LLMClient(settings=s)

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=None,
                    tool_calls=[
                        MagicMock(
                            id="call_abc",
                            function=MagicMock(name="write_file", arguments='{"path":"/tmp/x","content":"hi"}')
                        )
                    ]
                )
            )
        ]

        client._openai = MagicMock()
        client._openai.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.complete_with_tools(prompt="Write a file", model="gpt-4")

        assert isinstance(result, LLMResult)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "write_file"
        assert result.tool_calls[0].input == {"path": "/tmp/x", "content": "hi"}

    @pytest.mark.asyncio
    async def test_complete_simple(self):
        """The simple complete() method returns text only."""
        s = Settings(provider=Provider.ANTHROPIC, model="claude-3-5-haiku-20241022")
        client = LLMClient(settings=s)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Plain text response")]

        client._anthropic = MagicMock()
        client._anthropic.messages.create = AsyncMock(return_value=mock_response)

        text = await client.complete(prompt="Hello", model="claude-3-5-haiku-20241022")
        assert text == "Plain text response"

    @pytest.mark.asyncio
    async def test_complete_with_tools_passes_tools_to_anthropic(self):
        """Verify that tools are forwarded to the Anthropic API."""
        s = Settings(provider=Provider.ANTHROPIC, model="claude-3-5-sonnet-20241022")
        client = LLMClient(settings=s)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="ok")]

        client._anthropic = MagicMock()
        client._anthropic.messages.create = AsyncMock(return_value=mock_response)

        tool_defs = [{"name": "test_tool", "description": "test", "input_schema": {"type": "object"}}]
        await client.complete_with_tools(
            prompt="test", model="claude-3-5-sonnet-20241022",
            tools=tool_defs,
        )

        call_kwargs = client._anthropic.messages.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tool_defs

    @pytest.mark.asyncio
    async def test_complete_with_messages_continues_conversation(self):
        """When messages are provided, prompt is not duplicated."""
        s = Settings(provider=Provider.ANTHROPIC, model="claude-3-5-haiku-20241022")
        client = LLMClient(settings=s)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="ok")]

        client._anthropic = MagicMock()
        client._anthropic.messages.create = AsyncMock(return_value=mock_response)

        existing_messages = [
            {"role": "user", "content": [{"type": "text", "text": "previous"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
            {"role": "user", "content": [{"type": "text", "text": "more"}]},
        ]
        await client.complete_with_tools(
            prompt="", system="", model="claude-3-5-haiku-20241022",
            messages=existing_messages,
        )

        call_kwargs = client._anthropic.messages.create.call_args.kwargs
        # Should use existing messages, not add a new prompt
        assert len(call_kwargs["messages"]) == 3


class TestMessageConversion:
    def test_anthropic_to_openai_user_text(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result = LLMClient._anthropic_to_openai_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"

    def test_anthropic_to_openai_assistant_with_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll read that file."},
                    {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "/tmp/test"}},
                ],
            }
        ]
        result = LLMClient._anthropic_to_openai_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "I'll read that file."
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "read_file"

    def test_anthropic_to_openai_tool_result(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here's the result."},
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "file contents here"},
                ],
            }
        ]
        result = LLMClient._anthropic_to_openai_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert "Here's the result." in result[0]["content"]
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_1"

    def test_anthropic_to_openai_empty_assistant_content(self):
        """Assistant with only tool calls (no text) should have content=None."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_1", "name": "do_thing", "input": {}},
                ],
            }
        ]
        result = LLMClient._anthropic_to_openai_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] is None

    def test_anthropic_to_openai_tools(self):
        """Convert Anthropic tool definitions to OpenAI format."""
        anthropic_tools = [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            }
        ]
        result = LLMClient._anthropic_to_openai_tools(anthropic_tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "read_file"
        assert result[0]["function"]["description"] == "Read a file"
        assert result[0]["function"]["parameters"] == anthropic_tools[0]["input_schema"]

    def test_anthropic_to_openai_tools_multiple(self):
        """Convert multiple Anthropic tool definitions."""
        anthropic_tools = [
            {"name": "tool1", "description": "First", "input_schema": {"type": "object"}},
            {"name": "tool2", "description": "Second", "input_schema": {"type": "object"}},
        ]
        result = LLMClient._anthropic_to_openai_tools(anthropic_tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool1"
        assert result[1]["function"]["name"] == "tool2"
