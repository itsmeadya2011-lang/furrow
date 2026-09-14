import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient


class TestPlannerAgentPlan:
    @pytest.mark.asyncio
    async def test_plan_parses_valid_json(self):
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "tasks": [
                {"id": "1", "description": "task 1", "files": [], "dependencies": [], "status": "pending"},
                {"id": "2", "description": "task 2", "files": ["src/a.py"], "dependencies": [], "status": "pending"},
            ],
            "rationale": "we need to do these things",
        }))
        mock_client.settings.planner_model = "test-planner"

        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("build an API")

        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 2
        assert plan.tasks[0].id == "1"
        assert plan.tasks[0].description == "task 1"
        assert plan.tasks[1].id == "2"
        assert plan.tasks[1].description == "task 2"
        assert plan.rationale == "we need to do these things"

    @pytest.mark.asyncio
    async def test_plan_minimal_task(self):
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "tasks": [{"id": "1", "description": "do thing"}],
            "rationale": "simple",
        }))
        mock_client.settings.planner_model = "test-planner"

        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("do thing")

        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "1"
        assert plan.tasks[0].description == "do thing"
        assert plan.tasks[0].files == []
        assert plan.tasks[0].dependencies == []
        assert plan.tasks[0].status == "pending"

    @pytest.mark.asyncio
    async def test_plan_raises_valueerror_on_invalid_json(self):
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value="this is not json")
        mock_client.settings.planner_model = "test-planner"

        planner = PlannerAgent(client=mock_client)

        with pytest.raises(ValueError, match="Failed to parse plan"):
            await planner.plan("test goal")

    @pytest.mark.asyncio
    async def test_plan_raises_valueerror_on_empty_response(self):
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value="")
        mock_client.settings.planner_model = "test-planner"

        planner = PlannerAgent(client=mock_client)

        with pytest.raises(ValueError):
            await planner.plan("test goal")

    @pytest.mark.asyncio
    async def test_plan_raises_valueerror_on_malformed_json(self):
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value="{not valid json")
        mock_client.settings.planner_model = "test-planner"

        planner = PlannerAgent(client=mock_client)

        with pytest.raises(ValueError):
            await planner.plan("test goal")

    @pytest.mark.asyncio
    async def test_plan_calls_complete_with_correct_prompt(self):
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "tasks": [],
            "rationale": "no tasks",
        }))
        mock_client.settings.planner_model = "test-planner"

        planner = PlannerAgent(client=mock_client)
        await planner.plan("my goal")

        call_args = mock_client.complete.call_args
        assert call_args is not None
        prompt = call_args.args[0]
        assert "my goal" in prompt
        mock_client.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_plan_uses_planner_model(self):
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "tasks": [],
            "rationale": "no tasks",
        }))
        mock_client.settings.planner_model = "special-planner-model"

        planner = PlannerAgent(client=mock_client)
        await planner.plan("test")

        call_kwargs = mock_client.complete.call_args.kwargs
        assert call_kwargs.get("model") == "special-planner-model"
