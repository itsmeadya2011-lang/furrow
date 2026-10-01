"""Tests for the PlannerAgent — JSON parsing and parallelism limits."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.planner import PlannerAgent


@pytest.fixture
def client():
    c = MagicMock()
    c.settings.planner_model = "claude-3-5-haiku"
    c.settings.max_parallel_tasks = 3
    c.complete = AsyncMock()
    return c


@pytest.mark.asyncio
async def test_plan_parses_plain_json(client):
    plan_data = {
        "tasks": [
            {"id": "1", "description": "Task 1"},
            {"id": "2", "description": "Task 2"},
        ],
        "rationale": "Simple plan",
    }
    client.complete.return_value = json.dumps(plan_data)

    planner = PlannerAgent(client=client)
    plan = await planner.plan("Make a website")

    assert len(plan.tasks) == 2
    assert plan.rationale == "Simple plan"
    assert plan.tasks[0].description == "Task 1"


@pytest.mark.asyncio
async def test_plan_parses_markdown_fenced_json(client):
    plan_data = {
        "tasks": [{"id": "1", "description": "Do something"}],
        "rationale": "OK",
    }
    response = "```json\n" + json.dumps(plan_data) + "\n```"
    client.complete.return_value = response

    planner = PlannerAgent(client=client)
    plan = await planner.plan("Fix bug")

    assert plan.tasks[0].description == "Do something"


@pytest.mark.asyncio
async def test_plan_truncates_to_max_parallel(client):
    tasks = [{"id": str(i), "description": f"Task {i}"} for i in range(10)]
    plan_data = {"tasks": tasks, "rationale": "Too many tasks"}
    client.complete.return_value = json.dumps(plan_data)

    planner = PlannerAgent(client=client)
    plan = await planner.plan("Big project")

    assert len(plan.tasks) == 3  # max_parallel_tasks is 3


@pytest.mark.asyncio
async def test_plan_parse_fallback_finds_json_in_text(client):
    """If LLM adds preamble text, parser should still find the JSON object."""
    plan_data = {
        "tasks": [{"id": "1", "description": "Task 1", "files": [], "dependencies": [], "status": "pending"}],
        "rationale": "OK",
    }
    response = "Here is my plan:\n" + json.dumps(plan_data) + "\nThat's it."
    client.complete.return_value = response

    planner = PlannerAgent(client=client)
    plan = await planner.plan("Some goal")

    assert plan.tasks[0].description == "Task 1"


@pytest.mark.asyncio
async def test_plan_raises_on_invalid_json(client):
    client.complete.return_value = "This is not JSON at all."

    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("Some goal")
