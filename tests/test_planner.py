from __future__ import annotations

import json

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan
from tests.conftest import MockLLMClient


def _planner(responses: list[str]) -> PlannerAgent:
    return PlannerAgent(client=MockLLMClient(responses=responses))


class TestPlanner:
    async def test_plan_with_valid_json(self):
        resp = json.dumps(
            {
                "tasks": [
                    {
                        "id": "1",
                        "description": "do thing",
                        "files": ["a.py"],
                        "dependencies": [],
                    }
                ],
                "rationale": "plan it",
            }
        )
        planner = _planner([resp])
        plan = await planner.plan("build thing")
        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "do thing"
        assert plan.rationale == "plan it"

    async def test_plan_empty_tasks(self):
        resp = json.dumps({"tasks": [], "rationale": "already done"})
        planner = _planner([resp])
        plan = await planner.plan("noop")
        assert plan.tasks == []

    async def test_plan_invalid_json_raises(self):
        planner = _planner(["not json"])
        with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
            await planner.plan("build")

    async def test_plan_missing_fields_raises(self):
        resp = json.dumps({"rationale": "no tasks field"})
        planner = _planner([resp])
        with pytest.raises(ValueError):
            await planner.plan("build")

    async def test_plan_uses_planner_model(self):
        resp = json.dumps({"tasks": [], "rationale": "ok"})
        client = MockLLMClient(responses=[resp])
        planner = PlannerAgent(client=client)
        await planner.plan("goal")
        last = client.calls[-1]
        assert last["model"] == client.settings.planner_model

    async def test_plan_passes_goal_in_prompt(self):
        resp = json.dumps({"tasks": [], "rationale": "ok"})
        client = MockLLMClient(responses=[resp])
        planner = PlannerAgent(client=client)
        await planner.plan("a special goal")
        assert "a special goal" in client.calls[-1]["prompt"]
