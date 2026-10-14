import json
from unittest.mock import AsyncMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, Settings
from furrow.llm import LLMClient


class TestPlannerAgent:
    @pytest.fixture
    def mock_client(self):
        settings = Settings(anthropic_api_key="test-key")
        client = LLMClient(settings=settings)
        client.complete = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_plan_returns_valid_plan(self, mock_client):
        plan_data = {
            "tasks": [
                {"id": "1", "description": "Task 1", "files": ["file1.py"]},
                {"id": "2", "description": "Task 2", "files": ["file2.py"], "dependencies": ["1"]},
            ],
            "rationale": "Test rationale",
        }
        mock_client.complete.return_value = json.dumps(plan_data)

        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("Test goal")

        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 2
        assert plan.tasks[0].id == "1"
        assert plan.tasks[0].description == "Task 1"
        assert plan.tasks[1].dependencies == ["1"]
        assert plan.rationale == "Test rationale"

    @pytest.mark.asyncio
    async def test_json_parsing_of_plan_response(self, mock_client):
        raw_response = """
        Here is the plan:
        {"tasks": [{"id": "1", "description": "Do something"}], "rationale": "reason"}
        """
        mock_client.complete.return_value = raw_response

        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("Test goal")

        assert isinstance(plan, Plan)
        assert plan.tasks[0].id == "1"

    @pytest.mark.asyncio
    async def test_retry_on_parse_failure(self, mock_client):
        mock_client.complete.side_effect = [
            '{"invalid": json}',
            '{"tasks": [{"id": "1", "description": "Valid"}], "rationale": "ok"}',
        ]

        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("Test goal")

        assert mock_client.complete.call_count == 2
        assert plan.tasks[0].description == "Valid"

    @pytest.mark.asyncio
    async def test_workspace_context_included_in_prompt(self, mock_client, tmp_path):
        plan_data = {"tasks": [{"id": "1", "description": "Task"}], "rationale": "test"}
        mock_client.complete.return_value = json.dumps(plan_data)

        settings = Settings(workspace=tmp_path, anthropic_api_key="test-key")
        client = LLMClient(settings=settings)
        client.complete = mock_client.complete

        (tmp_path / "existing.py").write_text("print('hello')")

        planner = PlannerAgent(client=client)
        await planner.plan("Test goal")

        call_args = client.complete.call_args[0][0]
        assert "existing.py" in call_args
        assert str(tmp_path) in call_args

    @pytest.mark.asyncio
    async def test_error_handling_on_invalid_response(self, mock_client):
        mock_client.complete.return_value = "not json at all"

        planner = PlannerAgent(client=mock_client)

        with pytest.raises(ValueError, match="Failed to parse a valid plan"):
            await planner.plan("Test goal")

        assert mock_client.complete.call_count == 3  # MAX_PLAN_ATTEMPTS

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_error(self, mock_client):
        mock_client.complete.return_value = "invalid json"

        planner = PlannerAgent(client=mock_client)

        with pytest.raises(ValueError, match="Failed to parse a valid plan"):
            await planner.plan("Test goal")

    @pytest.mark.asyncio
    async def test_missing_tasks_field_raises_error(self, mock_client):
        mock_client.complete.return_value = '{"rationale": "no tasks field"}'

        planner = PlannerAgent(client=mock_client)

        with pytest.raises(ValueError, match="missing required 'tasks' field"):
            await planner.plan("Test goal")

    @pytest.mark.asyncio
    async def test_plan_with_empty_tasks(self, mock_client):
        plan_data = {"tasks": [], "rationale": "nothing to do"}
        mock_client.complete.return_value = json.dumps(plan_data)

        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("Test goal")

        assert plan.tasks == []
        assert plan.rationale == "nothing to do"

    @pytest.mark.asyncio
    async def test_planner_uses_planner_model(self, mock_client):
        plan_data = {"tasks": [{"id": "1", "description": "Task"}], "rationale": "test"}
        mock_client.complete.return_value = json.dumps(plan_data)

        settings = Settings(planner_model="custom-planner-model", anthropic_api_key="test-key")
        client = LLMClient(settings=settings)
        client.complete = mock_client.complete

        planner = PlannerAgent(client=client)
        await planner.plan("Test goal")

        assert client.complete.call_args[1]["model"] == "custom-planner-model"