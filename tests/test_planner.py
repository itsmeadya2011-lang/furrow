import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from furrow.agents.planner import PlannerAgent
from furrow.config import Plan

class TestPlannerParse:
    @pytest.mark.asyncio
    async def test_parses_valid_json_plan(self):
        client = MagicMock()
        valid_json = json.dumps({
            "tasks": [{"id": "1", "description": "do thing", "files": [], "dependencies": []}],
            "rationale": "simple plan"
        })
        client.complete = AsyncMock(return_value=valid_json)
        client.settings = MagicMock()
        client.settings.planner_model = "test-model"

        planner = PlannerAgent(client=client)
        plan = await planner.plan("test goal")

        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "do thing"

    @pytest.mark.asyncio
    async def test_raises_on_invalid_json(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value="not json at all")
        client.settings = MagicMock()
        client.settings.planner_model = "test-model"

        planner = PlannerAgent(client=client)

        with pytest.raises(ValueError, match="Failed to parse plan"):
            await planner.plan("test goal")