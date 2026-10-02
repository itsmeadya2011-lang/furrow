"""Tests for furrow.agents.planner: PlannerAgent.plan JSON parsing and errors."""

import json
from unittest.mock import AsyncMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, Provider, Settings


class MockLLMClient:
    """Minimal double for LLMClient used by agents."""

    def __init__(self, settings: Settings, response: str) -> None:
        self.settings = settings
        self.complete = AsyncMock(return_value=response)


def make_client(response: str) -> "MockLLMClient":
    settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="sk-test")
    return MockLLMClient(settings=settings, response=response)


async def test_plan_parses_valid_json_response():
    data = {
        "tasks": [
            {"id": "1", "description": "impl feature", "files": ["a.py"], "dependencies": []},
            {"id": "2", "description": "add tests", "files": ["tests/test_a.py"], "dependencies": ["1"]},
        ],
        "rationale": "implement and test",
    }
    client = make_client(json.dumps(data))
    planner = PlannerAgent(client=client)
    plan = await planner.plan("build feature")
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 2
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "impl feature"
    assert plan.tasks[0].files == ["a.py"]
    assert plan.tasks[1].dependencies == ["1"]
    assert plan.rationale == "implement and test"


async def test_plan_calls_complete_with_model():
    client = make_client(json.dumps({"tasks": [], "rationale": "empty"}))
    planner = PlannerAgent(client=client)
    await planner.plan("do thing")
    client.complete.assert_awaited_once()
    kwargs = client.complete.await_args.kwargs
    assert kwargs["model"] == client.settings.planner_model


async def test_plan_raises_value_error_on_invalid_json():
    client = make_client("this is not valid json {{{")
    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
        await planner.plan("goal")


async def test_plan_raises_value_error_on_missing_required_field():
    client = make_client(json.dumps({"rationale": "missing tasks"}))
    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError):
        await planner.plan("goal")


async def test_plan_with_empty_tasks():
    client = make_client(json.dumps({"tasks": [], "rationale": "already done"}))
    planner = PlannerAgent(client=client)
    plan = await planner.plan("noop")
    assert plan.tasks == []
    assert plan.rationale == "already done"
