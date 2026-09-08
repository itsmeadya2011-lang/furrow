import json
from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


class TestLLMClientOllama:
    @pytest.mark.asyncio
    async def test_ollama_payload_construction(self):
        settings = Settings(provider=Provider.OLLAMA, model="llama3", ollama_base_url="http://localhost:11434")
        client = LLMClient(settings=settings)

        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        mock_response.json.return_value = {"message": {"content": "hello"}}

        mock_post = AsyncMock(return_value=mock_response)
        client._ollama_client = AsyncMock()
        client._ollama_client.post = mock_post

        result = await client._complete_ollama("hi", "sys", "llama3", 512)

        assert result == "hello"
        called_payload = mock_post.call_args[1]["json"]
        assert called_payload["model"] == "llama3"
        assert called_payload["stream"] is False
        assert called_payload["options"]["num_predict"] == 512
        messages = called_payload["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "sys"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_ollama_default_system(self):
        settings = Settings(provider=Provider.OLLAMA, model="llama3")
        client = LLMClient(settings=settings)

        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        mock_response.json.return_value = {"message": {"content": "hi"}}

        mock_post = AsyncMock(return_value=mock_response)
        client._ollama_client = AsyncMock()
        client._ollama_client.post = mock_post

        await client._complete_ollama("prompt", "", "llama3", 512)
        called_payload = mock_post.call_args[1]["json"]
        assert called_payload["messages"][0]["content"] == "You are a helpful coding assistant."


class TestLLMClientRetry:
    @pytest.mark.asyncio
    async def test_retry_on_failure_then_success(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="sk-test")
        client = LLMClient(settings=settings)

        mock_response = AsyncMock()
        mock_response.content = [AsyncMock(text="success")]

        fail_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal fail_count
            fail_count += 1
            if fail_count < 3:
                raise Exception("transient error")
            return mock_response

        with patch.object(client, "anthropic") as mock_anthropic:
            mock_anthropic.messages.create = AsyncMock(side_effect=side_effect)
            result = await client._complete_anthropic("prompt", "sys", "model", 512)

        assert result == "success"
        assert fail_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="sk-test")
        client = LLMClient(settings=settings)

        with patch.object(client, "anthropic") as mock_anthropic:
            mock_anthropic.messages.create = AsyncMock(side_effect=Exception("always fails"))
            with pytest.raises(Exception):
                await client._complete_anthropic("prompt", "sys", "model", 512)


class TestLLMClientRouting:
    @pytest.mark.asyncio
    async def test_anthropic_routing(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="sk-test")
        client = LLMClient(settings=settings)

        mock_response = AsyncMock()
        mock_response.content = [AsyncMock(text="anthropic result")]

        with patch.object(client, "anthropic") as mock_anthropic:
            mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
            result = await client.complete("prompt", system="sys", model="claude")

        assert result == "anthropic result"

    @pytest.mark.asyncio
    async def test_openai_routing(self):
        settings = Settings(provider=Provider.OPENAI, openai_api_key="sk-test")
        client = LLMClient(settings=settings)

        mock_message = AsyncMock()
        mock_message.content = "openai result"
        mock_choice = AsyncMock(message=mock_message)

        with patch.object(client, "openai") as mock_openai:
            mock_openai.chat.completions.create = AsyncMock(return_value=AsyncMock(choices=[mock_choice]))
            result = await client.complete("prompt", system="sys", model="gpt")

        assert result == "openai result"
