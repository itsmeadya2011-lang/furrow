"""Tests for PlannerAgent — JSON parsing and error handling."""
from __future__ import annotations

import json

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, TaskModel

from tests.conftest import FakeLLMClient, fake_settings  # noqa: F401


class TestPlannerParse:
    async def test_plan_valid_json(self, fake_client):
        payload = json.dumps({
            "tasks": [
                {"id": "1", "description": "task1", "files": ["a.py"], "dependencies": []},
                {"id": "2", "description": "task2", "files": ["b.py"], "dependencies": ["1"]},
            ],
            "rationale": "two parallel tasks",
        })
        fake_client._response = payload

        planner = PlannerAgent(client=fake_client)
        plan = await planner.plan("build something")

        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 2
        assert plan.tasks[0].id == "1"
        assert plan.tasks[0].description == "task1"
        assert plan.tasks[1].dependencies == ["1"]
        assert plan.rationale == "two parallel tasks"

    async def test_plan_single_task(self, fake_client):
        payload = json.dumps({
            "tasks": [{"id": "1", "description": "do the thing", "files": [], "dependencies": []}],
            "rationale": "simple",
        })
        fake_client._response = payload

        planner = PlannerAgent(client=fake_client)
        plan = await planner.plan("simple goal")

        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "do the thing"

    async def test_plan_invalid_json_raises(self, fake_client):
        fake_client._response = "this is not valid JSON {{{"

        planner = PlannerAgent(client=fake_client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await planner.plan("goal")

    async def test_plan_missing_required_fields_raises(self, fake_client):
        fake_client._response = json.dumps({"rationale": "no tasks"})

        from pydantic import ValidationError

        planner = PlannerAgent(client=fake_client)
        with pytest.raises((ValueError, ValidationError)):
            await planner.plan("goal")

    async def test_plan_includes_goal_in_prompt(self, fake_client):
        fake_client._response = json.dumps({"tasks": [], "rationale": "ok"})

        planner = PlannerAgent(client=fake_client)
        await planner.plan("my specific goal text")

        assert len(fake_client.complete_calls) == 1
        prompt = fake_client.complete_calls[0][0]
        assert "my specific goal text" in prompt

    async def test_plan_uses_planner_model(self, fake_client):
        fake_client._response = json.dumps({"tasks": [], "rationale": "ok"})

        planner = PlannerAgent(client=fake_client)
        await planner.plan("goal")

        model = fake_client.complete_calls[0][2]
        assert model == fake_client.settings.planner_model
