from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, Settings


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_invalid_json_raises(self):
        client = MagicMock()
        client.settings = Settings()
        client.complete = AsyncMock(return_value="not valid json")
        agent = PlannerAgent(client=client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await agent.plan("do something")

    @pytest.mark.asyncio
    async def test_valid_json_returns_plan(self):
        client = MagicMock()
        client.settings = Settings()
        data = {
            "tasks": [{"id": "1", "description": "do thing", "files": [], "dependencies": []}],
            "rationale": "ok",
        }
        client.complete = AsyncMock(return_value=json.dumps(data))
        agent = PlannerAgent(client=client)
        plan = await agent.plan("do something")
        assert isinstance(plan, Plan)
        assert plan.tasks[0].description == "do thing"
