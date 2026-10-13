"""Shared test fixtures for Furrow tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest


class MockLLMClient:
    """Reusable mock LLM client for tests."""

    def __init__(self, response="", responses=None):
        self.response = response
        self.responses = responses or []
        self.call_count = 0
        self.settings = MagicMock()
        self.settings.planner_model = "test-planner"
        self.settings.worker_model = "test-worker"
        self.settings.tester_model = "test-tester"
        self.settings.model = "test-model"
        self.settings.provider = MagicMock()
        self.settings.ollama_base_url = "http://localhost:11434"
        self.settings.max_parallel_tasks = 5
        self.settings.max_cycles = 0

    async def complete(self, prompt, system="", model=None):
        self.call_count += 1
        if self.responses and self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]
        return self.response

    async def aclose(self):
        pass


@pytest.fixture
def mock_llm():
    return MockLLMClient()
