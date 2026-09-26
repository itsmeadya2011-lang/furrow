from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


class MockSettings:
    def __init__(self, **kwargs) -> None:
        self.max_cycles = kwargs.get("max_cycles", 0)
        self.max_parallel_tasks = kwargs.get("max_parallel_tasks", 5)
        self.provider = kwargs.get("provider")
        self.model = kwargs.get("model", "gpt-4o")
        self.planner_model = kwargs.get("planner_model", "gpt-4o-mini")
        self.worker_model = kwargs.get("worker_model", "gpt-4o")
        self.tester_model = kwargs.get("tester_model", "gpt-4o")
        self.ollama_base_url = kwargs.get("ollama_base_url", "http://localhost:11434")
        self.anthropic_api_key = kwargs.get("anthropic_api_key")
        self.openai_api_key = kwargs.get("openai_api_key")


class MockClient:
    def __init__(self, settings=None, **kwargs) -> None:
        self.settings = settings if settings is not None else MockSettings(**kwargs)
        self.complete = AsyncMock()
        self.read_file = AsyncMock()
        self.write_file = AsyncMock()
        self.list_files = MagicMock()


@pytest.fixture
def mock_client() -> MockClient:
    return MockClient()
