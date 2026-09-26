from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

SAMPLE_PLAN_JSON = """
{
    "tasks": [
        {
            "id": "1",
            "description": "Implement feature A",
            "files": ["src/feature_a.py"],
            "dependencies": []
        },
        {
            "id": "2",
            "description": "Add tests for feature A",
            "files": ["tests/test_feature_a.py"],
            "dependencies": ["1"]
        }
    ],
    "rationale": "Break the goal into implementation and testing tasks."
}
"""


@pytest.fixture
def sample_plan_json() -> str:
    return SAMPLE_PLAN_JSON


@pytest.fixture
def mock_llm_client() -> MagicMock:
    client = MagicMock()
    client.complete = AsyncMock(return_value=SAMPLE_PLAN_JSON)
    client.settings = MagicMock()
    client.settings.model = "claude-sonnet-4-20250514"
    client.settings.planner_model = "claude-3-5-haiku-20241022"
    client.settings.worker_model = "claude-3-5-sonnet-20241022"
    client.settings.tester_model = "claude-3-5-sonnet-20241022"
    client.settings.ollama_model = "llama3"
    client.settings.ollama_base_url = "http://localhost:11434"
    client.settings.provider = MagicMock()
    return client
