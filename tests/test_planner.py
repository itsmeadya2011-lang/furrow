import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan


@pytest.mark.asyncio
async def test_plan_valid_json():
    client = MagicMock()
    client.complete = AsyncMock(return_value=json.dumps({"tasks": [{"id": "1", "description": "d"}], "rationale": "r"}))
    client.settings = MagicMock()
    client.settings.planner_model = "m"
    planner = PlannerAgent(client=client)
    plan = await planner.plan("goal")
    assert isinstance(plan, Plan)
    assert plan.tasks[0].id == "1"


@pytest.mark.asyncio
async def test_plan_invalid_json():
    client = MagicMock()
    client.complete = AsyncMock(return_value="not json")
    client.settings = MagicMock()
    client.settings.planner_model = "m"
    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError):
        await planner.plan("goal")
