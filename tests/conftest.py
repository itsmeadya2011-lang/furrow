from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        provider=Provider.ANTHROPIC,
        model="claude-3-5-sonnet-20241022",
        planner_model="claude-3-5-haiku-20241022",
        worker_model="claude-3-5-sonnet-20241022",
        tester_model="claude-3-5-sonnet-20241022",
        max_parallel_tasks=5,
        max_cycles=0,
    )


@pytest.fixture
def mock_llm_client(mock_settings: Settings) -> LLMClient:
    client = LLMClient(mock_settings)
    client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "test"}')
    client.read_file = AsyncMock(return_value="")
    client.write_file = AsyncMock()
    return client
