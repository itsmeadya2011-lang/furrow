"""Tests for the PlannerAgent, including retry logic and error handling."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel
from furrow.agents.planner import PlannerAgent


class MockLLMClient:
    def __init__(self, response=""):
        self.settings = MagicMock()
        self.settings.planner_model = "test-model"
        self._response = response
        self.call_count = 0

    async def complete(self, prompt, system="", model=None):
        self.call_count += 1
        return self._response


@pytest.mark.asyncio
async def test_plan_parse_valid_json():
    """Planner should parse valid JSON response into a Plan."""
    response = json.dumps({
        "tasks": [{"id": "1", "description": "do thing", "files": ["f.py"], "dependencies": []}],
        "rationale": "test plan"
    })
    client = MockLLMClient(response)
    planner = PlannerAgent(client=client)
    plan = await planner.plan("test goal")
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "do thing"
    assert plan.rationale == "test plan"


@pytest.mark.asyncio
async def test_plan_parse_invalid_json_retries():
    """Planner should retry when JSON is invalid, up to max_retries."""
    # First response is invalid, second is valid
    responses = [
        "not json at all",
        '{"tasks": [{"id": "1", "description": "ok", "dependencies": []}], "rationale": "ok"}',
    ]
    client = MockLLMClient("")
    call_count = 0

    async def mock_complete(prompt, system="", model=None):
        nonlocal call_count
        call_count += 1
        return responses[call_count - 1]

    client.complete = mock_complete
    planner = PlannerAgent(client=client)
    plan = await planner.plan("test goal", max_retries=3)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "ok"
    assert call_count == 2  # Should have retried


@pytest.mark.asyncio
async def test_plan_parse_all_failures_returns_empty():
    """Planner should return empty plan when all retries fail."""
    client = MockLLMClient("not json")
    planner = PlannerAgent(client=client)
    plan = await planner.plan("test goal", max_retries=3)
    assert len(plan.tasks) == 0
    assert "Failed" in plan.rationale


@pytest.mark.asyncio
async def test_plan_parse_json_with_markdown_wrapping():
    """Planner should retry if response is JSON wrapped in markdown."""
    response = "```json\n{\"tasks\": [], \"rationale\": \"empty\"}\n```"
    client = MockLLMClient(response)
    planner = PlannerAgent(client=client)
    plan = await planner.plan("test goal", max_retries=2)
    assert len(plan.tasks) == 0
    # It should have retried since JSON.parse fails on the markdown
    assert client.call_count >= 2


@pytest.mark.asyncio
async def test_plan_empty_goal():
    """Planner should handle empty goals gracefully."""
    response = '{"tasks": [], "rationale": "nothing to do"}'
    client = MockLLMClient(response)
    planner = PlannerAgent(client=client)
    plan = await planner.plan("")
    assert len(plan.tasks) == 0


@pytest.mark.asyncio
async def test_plan_stores_client_settings():
    """Planner should use client's settings for model selection."""
    client = MockLLMClient('{"tasks": [], "rationale": "ok"}')
    planner = PlannerAgent(client=client)
    await planner.plan("test")
    assert client.settings.planner_model == "test-model"
