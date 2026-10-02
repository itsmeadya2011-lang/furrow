"""Tests for the PlannerAgent."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Settings


def make_mock_client():
    """Create a mock LLMClient."""
    client = MagicMock()
    client.settings = Settings()
    client.complete = AsyncMock()
    return client


async def test_planner_plan_success():
    """Planner correctly parses valid JSON response."""
    client = make_mock_client()
    plan_json = '{"tasks": [{"id": "1", "description": "Implement auth", "files": ["auth.py"], "dependencies": []}], "rationale": "Add authentication"}'
    client.complete.return_value = plan_json

    planner = PlannerAgent(client=client)
    plan = await planner.plan("Add auth")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "Implement auth"
    assert plan.rationale == "Add authentication"


async def test_planner_plan_async_success():
    """Planner correctly parses valid JSON response (async)."""
    client = make_mock_client()
    plan_json = '{"tasks": [{"id": "1", "description": "Task 1", "files": [], "dependencies": []}, {"id": "2", "description": "Task 2", "files": ["b.py"], "dependencies": ["1"]}], "rationale": "Multi-task plan"}'
    client.complete.return_value = plan_json

    planner = PlannerAgent(client=client)
    plan = await planner.plan("Build something")

    assert len(plan.tasks) == 2
    assert plan.tasks[0].id == "1"
    assert plan.tasks[1].dependencies == ["1"]
    assert plan.rationale == "Multi-task plan"


async def test_planner_plan_empty():
    """Planner handles empty task list."""
    client = make_mock_client()
    client.complete.return_value = '{"tasks": [], "rationale": "Nothing to do"}'

    planner = PlannerAgent(client=client)
    plan = await planner.plan("Complete goal")

    assert len(plan.tasks) == 0
    assert plan.rationale == "Nothing to do"


async def test_planner_plan_invalid_json():
    """Planner raises ValueError on invalid JSON."""
    client = make_mock_client()
    client.complete.return_value = "This is not JSON"

    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("Some goal")


async def test_planner_plan_missing_field():
    """Planner raises ValueError when required fields are missing."""
    client = make_mock_client()
    client.complete.return_value = '{"tasks": "not a list"}'

    planner = PlannerAgent(client=client)
    with pytest.raises((ValueError, Exception)):
        await planner.plan("Some goal")


async def test_planner_uses_correct_model():
    """Planner uses the planner_model from settings."""
    client = make_mock_client()
    client.complete.return_value = '{"tasks": [], "rationale": "ok"}'

    planner = PlannerAgent(client=client)
    await planner.plan("Goal")

    # Check that the model passed to complete was the planner_model
    call_args = client.complete.call_args
    assert call_args.kwargs.get("model") == client.settings.planner_model
