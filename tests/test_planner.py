import pytest
from unittest.mock import MagicMock, AsyncMock

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, TaskModel


@pytest.mark.asyncio
async def test_plan_success():
    client = MagicMock()
    client.complete = AsyncMock(return_value='{"tasks": [{"id": "1", "description": "test"}], "rationale": "ok"}')
    planner = PlannerAgent(client=client)
    plan = await planner.plan("test goal")
    assert plan.tasks[0].id == "1"
    assert plan.rationale == "ok"
    client.complete.assert_called_once()


@pytest.mark.asyncio
async def test_plan_retry_on_bad_json():
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        "bad json",
        '{"tasks": [{"id": "1", "description": "test"}], "rationale": "ok"}',
    ])
    planner = PlannerAgent(client=client)
    plan = await planner.plan("test goal")
    assert plan.tasks[0].id == "1"
    assert client.complete.call_count == 2


@pytest.mark.asyncio
async def test_plan_all_retries_fail():
    client = MagicMock()
    client.complete = AsyncMock(return_value="bad json")
    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError):
        await planner.plan("test goal")
    assert client.complete.call_count == 3
