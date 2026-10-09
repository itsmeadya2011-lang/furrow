from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, Settings
from furrow.llm import LLMClient


class TestPlannerAgent:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=LLMClient)
        client.settings = Settings(planner_model="test-model")
        return client

    @pytest.fixture
    def planner(self, mock_client):
        return PlannerAgent(client=mock_client)

    @pytest.mark.asyncio
    async def test_plan_parses_json_correctly(self, planner, mock_client):
        json_response = '''
        {
            "tasks": [
                {"id": "1", "description": "Create main module", "files": ["main.py"]},
                {"id": "2", "description": "Create tests", "files": ["test_main.py"], "dependencies": ["1"]}
            ],
            "rationale": "Need main module and tests"
        }
        '''
        mock_client.complete = AsyncMock(return_value=json_response)

        plan = await planner.plan("Create a main module with tests")

        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 2
        assert plan.tasks[0].id == "1"
        assert plan.tasks[0].description == "Create main module"
        assert plan.tasks[0].files == ["main.py"]
        assert plan.tasks[1].id == "2"
        assert plan.tasks[1].dependencies == ["1"]
        assert plan.rationale == "Need main module and tests"

    @pytest.mark.asyncio
    async def test_plan_parses_json_with_extra_whitespace(self, planner, mock_client):
        json_response = '  \n  {\n  "tasks": [{"id": "1", "description": "Task"}],\n  "rationale": "Reason"\n}  \n'
        mock_client.complete = AsyncMock(return_value=json_response)

        plan = await planner.plan("Test goal")

        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "1"

    @pytest.mark.asyncio
    async def test_plan_raises_value_error_on_invalid_json(self, planner, mock_client):
        mock_client.complete = AsyncMock(return_value="This is not valid JSON")

        with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
            await planner.plan("Test goal")

    @pytest.mark.asyncio
    async def test_plan_raises_value_error_on_missing_fields(self, planner, mock_client):
        json_response = '{"tasks": [], "rationale": "Reason"}'
        mock_client.complete = AsyncMock(return_value=json_response)

        plan = await planner.plan("Test goal")
        assert plan.tasks == []
        assert plan.rationale == "Reason"

    @pytest.mark.asyncio
    async def test_plan_raises_value_error_on_invalid_task_structure(self, planner, mock_client):
        json_response = '{"tasks": [{"id": "1"}], "rationale": "Reason"}'
        mock_client.complete = AsyncMock(return_value=json_response)

        with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
            await planner.plan("Test goal")

    @pytest.mark.asyncio
    async def test_plan_uses_correct_model(self, planner, mock_client):
        json_response = '{"tasks": [], "rationale": "Reason"}'
        mock_client.complete = AsyncMock(return_value=json_response)

        await planner.plan("Test goal")

        mock_client.complete.assert_called_once()
        call_kwargs = mock_client.complete.call_args.kwargs
        assert call_kwargs.get('model') == "test-model"

    @pytest.mark.asyncio
    async def test_plan_includes_goal_in_prompt(self, planner, mock_client):
        json_response = '{"tasks": [], "rationale": "Reason"}'
        mock_client.complete = AsyncMock(return_value=json_response)

        await planner.plan("Specific goal here")

        call_args = mock_client.complete.call_args
        assert "Specific goal here" in call_args.args[0]