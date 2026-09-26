"""Shared pytest fixtures for Furrow tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.config import Provider, Settings


@pytest.fixture
def mock_settings() -> Settings:
    """Settings with test-safe values (no real API calls)."""
    return Settings(
        provider=Provider.ANTHROPIC,
        model="claude-3-5-haiku-20241022",
        planner_model="claude-3-5-haiku-20241022",
        worker_model="claude-3-5-sonnet-20241022",
        tester_model="claude-3-5-sonnet-20241022",
        anthropic_api_key="test-anthropic-key",
        openai_api_key="test-openai-key",
        max_parallel_tasks=5,
        max_cycles=0,
    )


@pytest.fixture
def mock_llm_client(mock_settings: Settings) -> MagicMock:
    """A MagicMock standing in for LLMClient with an async complete()."""
    client = MagicMock()
    client.settings = mock_settings
    client.complete = AsyncMock()
    client.read_file = AsyncMock(return_value="")
    client.write_file = AsyncMock()
    client.list_files = MagicMock(return_value=[])
    return client


@pytest.fixture
def openai_settings() -> Settings:
    """Settings configured for OpenAI provider."""
    return Settings(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        planner_model="gpt-4o-mini",
        worker_model="gpt-4o",
        tester_model="gpt-4o",
        openai_api_key="test-openai-key",
        max_cycles=0,
    )


@pytest.fixture
def ollama_settings() -> Settings:
    """Settings configured for Ollama provider."""
    return Settings(
        provider=Provider.OLLAMA,
        model="llama3",
        planner_model="llama3",
        worker_model="llama3",
        tester_model="llama3",
        ollama_base_url="http://localhost:11434",
        max_cycles=0,
    )
