"""Tests for the planner agent."""

from __future__ import annotations

import json

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, Settings
from tests.helpers import FakeLLMClient, make_plan


async def test_planner_parses_a_fenced_plan(settings: Settings) -> None:
    payload = "```json\n" + json.dumps(
        {
            "tasks": [
                {"id": "a", "description": "first", "files": ["a.py"], "dependencies": []},
                {"id": "b", "description": "second", "files": [], "dependencies": ["a"]},
            ],
            "rationale": "do it in order",
        }
    ) + "\n```"
    client = FakeLLMClient(settings, plans=[payload])

    plan = await PlannerAgent(client=client).plan("build a thing")

    assert isinstance(plan, Plan)
    assert [task.id for task in plan.tasks] == ["a", "b"]
    assert [task.description for task in plan.tasks] == ["first", "second"]
    assert plan.tasks[0].files == ["a.py"]
    assert plan.tasks[1].dependencies == ["a"]
    assert plan.rationale == "do it in order"


async def test_planner_caps_the_plan_at_five_tasks(settings: Settings) -> None:
    client = FakeLLMClient(settings, plans=[make_plan("a", "b", "c", "d", "e", "f", "g")])

    plan = await PlannerAgent(client=client).plan("goal")

    assert len(plan.tasks) == 5
    assert [task.description for task in plan.tasks] == ["a", "b", "c", "d", "e"]


async def test_planner_assigns_sequential_ids_when_the_model_omits_them(settings: Settings) -> None:
    payload = json.dumps(
        {"tasks": [{"description": "one"}, {"description": "two"}], "rationale": "r"}
    )
    client = FakeLLMClient(settings, plans=[payload])

    plan = await PlannerAgent(client=client).plan("goal")

    assert [task.id for task in plan.tasks] == ["1", "2"]
    assert [task.description for task in plan.tasks] == ["one", "two"]


async def test_planner_rejects_a_malformed_response(settings: Settings) -> None:
    client = FakeLLMClient(settings, plans=["not json at all"])

    with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
        await PlannerAgent(client=client).plan("goal")


async def test_planner_rejects_a_plan_without_tasks(settings: Settings) -> None:
    client = FakeLLMClient(settings, plans=[json.dumps({"tasks": [], "rationale": "r"})])

    with pytest.raises(ValueError, match="Plan missing non-empty tasks"):
        await PlannerAgent(client=client).plan("goal")


async def test_planner_rejects_a_plan_whose_tasks_are_all_invalid(settings: Settings) -> None:
    payload = json.dumps({"tasks": [{"no_description": True}], "rationale": "r"})
    client = FakeLLMClient(settings, plans=[payload])

    with pytest.raises(ValueError, match="No valid tasks in plan"):
        await PlannerAgent(client=client).plan("goal")


async def test_planner_sends_the_goal_and_uses_the_planner_model(settings: Settings) -> None:
    client = FakeLLMClient(settings, plans=[make_plan("only task")])

    await PlannerAgent(client=client).plan("make it blue")

    call = client.calls[-1]
    assert "make it blue" in call["prompt"]
    assert call["model"] == settings.planner_model
