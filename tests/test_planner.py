from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan


@pytest.mark.asyncio
async def test_planner_parses_json_response() -> None:
    client = MagicMock()
    client.settings.planner_model = "test-model"
    client.complete = AsyncMock(return_value=(
        '{"tasks": [{"id": "1", "description": "do thing", "files": ["a.py"], "dependencies": []}], '
        '"rationale": "because"}'
    ))
    agent = PlannerAgent(client=client)
    plan = await agent.plan("do something")
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "do thing"
    assert plan.rationale == "because"


@pytest.mark.asyncio
async def test_planner_raises_on_invalid_json() -> None:
    client = MagicMock()
    client.settings.planner_model = "test-model"
    client.complete = AsyncMock(return_value="not json")
    agent = PlannerAgent(client=client)
    with pytest.raises(ValueError):
        await agent.plan("do something")


@pytest.mark.asyncio
async def test_planner_raises_on_missing_fields() -> None:
    client = MagicMock()
    client.settings.planner_model = "test-model"
    client.complete = AsyncMock(return_value='{"tasks": []}')
    agent = PlannerAgent(client=client)
    with pytest.raises(ValueError):
        await agent.plan("do something")