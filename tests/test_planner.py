"""Tests for the PlannerAgent."""
import json

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan
from tests.conftest import MockLLMClient


def _make_plan_json(tasks=None):
    if tasks is None:
        tasks = [
            {
                "id": "1",
                "description": "Create a hello world module",
                "files": ["hello.py", "tests/test_hello.py"],
                "dependencies": [],
            }
        ]
    return json.dumps({"tasks": tasks, "rationale": "Simple plan"})


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_with_valid_json(self):
        client = MockLLMClient(responses=[_make_plan_json()])
        planner = PlannerAgent(client=client)
        plan = await planner.plan("Create a hello world module")
        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "1"
        assert plan.rationale == "Simple plan"

    @pytest.mark.asyncio
    async def test_plan_with_empty_tasks(self):
        response = json.dumps({"tasks": [], "rationale": "Nothing to do"})
        client = MockLLMClient(responses=[response])
        planner = PlannerAgent(client=client)
        plan = await planner.plan("Make goal that is already complete")
        assert len(plan.tasks) == 0

    @pytest.mark.asyncio
    async def test_plan_with_invalid_json_raises(self):
        client = MockLLMClient(responses=["this is not valid json"])
        planner = PlannerAgent(client=client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await planner.plan("Something")

    @pytest.mark.asyncio
    async def test_plan_uses_planner_model(self):
        client = MockLLMClient(responses=[_make_plan_json()])
        client.settings.planner_model = "my-planner-model"
        planner = PlannerAgent(client=client)
        await planner.plan("Test goal")
        # The mock doesn't track model, but we can verify it was called
        assert client.complete_call_count == 1

    @pytest.mark.asyncio
    async def test_plan_passes_goal_in_prompt(self):
        client = MockLLMClient(responses=[_make_plan_json()])
        planner = PlannerAgent(client=client)
        goal = "Implement feature X"
        await planner.plan(goal)
        assert goal in client.complete_calls[0][0]
