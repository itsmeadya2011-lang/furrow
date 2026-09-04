from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.fixture
def settings() -> Settings:
    return Settings(provider=Provider.ANTHROPIC, model="test-model")


class TestLLMClientRouting:
    @pytest.mark.asyncio
    async def test_routes_to_anthropic(self, settings: Settings) -> None:
        client = LLMClient(settings=settings)
        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock, return_value="hi") as mock_anthropic:
            result = await client.complete("prompt")
            mock_anthropic.assert_awaited_once_with("prompt", "", "test-model")
            assert result == "hi"

    @pytest.mark.asyncio
    async def test_routes_to_openai(self, settings: Settings) -> None:
        settings.provider = Provider.OPENAI
        client = LLMClient(settings=settings)
        with patch.object(client, "_complete_openai", new_callable=AsyncMock, return_value="hi") as mock_openai:
            result = await client.complete("prompt")
            mock_openai.assert_awaited_once_with("prompt", "", "test-model")
            assert result == "hi"

    @pytest.mark.asyncio
    async def test_uses_provided_model(self, settings: Settings) -> None:
        client = LLMClient(settings=settings)
        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock, return_value="hi") as mock_anthropic:
            await client.complete("prompt", model="other-model")
            mock_anthropic.assert_awaited_once_with("prompt", "", "other-model")

    @pytest.mark.asyncio
    async def test_unsupported_provider_raises(self, settings: Settings) -> None:
        settings.provider = Provider.OLLAMA
        client = LLMClient(settings=settings)
        with pytest.raises(ValueError, match="Unsupported provider"):
            await client.complete("prompt")


class TestLLMClientFileIO:
    @pytest.mark.asyncio
    async def test_write_and_read_file(self, tmp_path: Path) -> None:
        client = LLMClient()
        target = tmp_path / "sub" / "file.txt"
        await client.write_file(target, "hello world")
        assert target.exists()
        content = await client.read_file(target)
        assert content == "hello world"

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        client = LLMClient()
        target = tmp_path / "a" / "b" / "c.txt"
        await client.write_file(target, "x")
        assert target.exists()

    @pytest.mark.asyncio
    async def test_read_missing_file_raises(self, tmp_path: Path) -> None:
        client = LLMClient()
        with pytest.raises(OSError):
            await client.read_file(tmp_path / "missing.txt")


class TestLLMClientLifecycle:
    @pytest.mark.asyncio
    async def test_aclose_noop_when_not_initialized(self, settings: Settings) -> None:
        client = LLMClient(settings=settings)
        # _anthropic and _openai are None, should not raise
        await client.aclose()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, settings: Settings) -> None:
        client = LLMClient(settings=settings)
        async with client as ctx:
            assert ctx is client
        # aclose should have been called without error

    @pytest.mark.asyncio
    async def test_aclose_closes_clients(self, settings: Settings) -> None:
        client = LLMClient(settings=settings)
        mock_anthropic = AsyncMock()
        mock_openai = AsyncMock()
        client._anthropic = mock_anthropic
        client._openai = mock_openai
        await client.aclose()
        mock_anthropic.aclose.assert_awaited_once()
        mock_openai.aclose.assert_awaited_once()