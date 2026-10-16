from unittest.mock import AsyncMock

import pytest

from furrow.config import Plan, Settings
from furrow.agents.planner import PlannerAgent
from furrow.llm import LLMClient


def _json_plan():
    return '{"tasks": [{"id": "1", "description": "implement feature", "files": ["a.py"], "dependencies": []}], "rationale": "needed"}'


async def test_valid_json_parsed_into_plan():
    client = LLMClient(settings=Settings())
    client.complete = AsyncMock(return_value=_json_plan())
    planner = PlannerAgent(client=client)
    plan = await planner.plan("implement feature")
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "implement feature"
    assert plan.tasks[0].files == ["a.py"]
    assert plan.rationale == "needed"


async def test_invalid_json_raises_value_error():
    client = LLMClient(settings=Settings())
    client.complete = AsyncMock(return_value="not valid json")
    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("goal")


async def test_planner_uses_correct_model():
    client = LLMClient(settings=Settings())
    client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "r"}')
    planner = PlannerAgent(client=client)
    await planner.plan("goal")
    _, kwargs = client.complete.call_args
    assert kwargs["model"] == client.settings.planner_model
