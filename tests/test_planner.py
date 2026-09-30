"""Tests for furrow.agents.planner — plan parsing from LLM output."""
from __future__ import annotations

import json

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, Settings


class _FakeClient:
    """Minimal stand-in for LLMClient that returns a canned plan JSON."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.settings = Settings()
        self.complete_calls: list[tuple[str, str, str | None]] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.complete_calls.append((prompt, system, model))
        return self._response


@pytest.fixture
def valid_plan_json() -> str:
    return json.dumps({
        "tasks": [
            {
                "id": "1",
                "description": "Create the API router",
                "files": ["src/api/router.py", "tests/test_router.py"],
                "dependencies": [],
            },
            {
                "id": "2",
                "description": "Add authentication middleware",
                "files": ["src/auth/middleware.py"],
                "dependencies": [],
            },
        ],
        "rationale": "Two independent tasks for the initial scaffold.",
    })


@pytest.mark.asyncio
async def test_plan_success(valid_plan_json):
    client = _FakeClient(valid_plan_json)
    planner = PlannerAgent(client=client)
    plan = await planner.plan("Build a web API")
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 2
    assert plan.tasks[0].id == "1"
    assert plan.tasks[1].dependencies == []
    assert plan.rationale == "Two independent tasks for the initial scaffold."


@pytest.mark.asyncio
async def test_plan_uses_planner_model(valid_plan_json):
    client = _FakeClient(valid_plan_json)
    planner = PlannerAgent(client=client)
    await planner.plan("some goal")
    assert client.complete_calls[0][2] == client.settings.planner_model


@pytest.mark.asyncio
async def test_plan_passes_goal_in_prompt(valid_plan_json):
    client = _FakeClient(valid_plan_json)
    planner = PlannerAgent(client=client)
    await planner.plan("My specific goal")
    assert "My specific goal" in client.complete_calls[0][0]


@pytest.mark.asyncio
async def test_plan_invalid_json_raises():
    client = _FakeClient("not json at all")
    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("anything")


@pytest.mark.asyncio
async def test_plan_empty_tasks_valid_json():
    """When the planner returns an empty task list, Plan should parse it.

    (Empty list is valid JSON; the orchestrator handles the business rule.)
    """
    client = _FakeClient(json.dumps({"tasks": [], "rationale": "no work"}))
    planner = PlannerAgent(client=client)
    plan = await planner.plan("goal")
    assert plan.tasks == []
    assert plan.rationale == "no work"
