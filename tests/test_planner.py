import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from furrow.agents.planner import PlannerAgent
from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient


class TestPlannerAgent:
    """Tests for the PlannerAgent class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock LLMClient."""
        client = MagicMock(spec=LLMClient)
        client.settings = MagicMock()
        client.settings.planner_model = "test-planner-model"
        client.complete = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_planner_parses_valid_json_plan(self, mock_client):
        """Test planner parses valid JSON plan."""
        response = json.dumps({
            "tasks": [
                {
                    "id": "1",
                    "description": "Implement authentication",
                    "files": ["src/auth.py"],
                    "dependencies": [],
                },
                {
                    "id": "2",
                    "description": "Add tests",
                    "files": ["tests/test_auth.py"],
                    "dependencies": ["1"],
                },
            ],
            "rationale": "Build auth system first, then test it",
        })
        mock_client.complete.return_value = response

        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("Create an auth system")

        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 2
        assert plan.tasks[0].id == "1"
        assert plan.tasks[0].description == "Implement authentication"
        assert plan.tasks[1].id == "2"
        assert plan.rationale == "Build auth system first, then test it"

    @pytest.mark.asyncio
    async def test_planner_raises_on_invalid_json(self, mock_client):
        """Test planner raises on invalid JSON."""
        mock_client.complete.return_value = "This is not JSON"

        planner = PlannerAgent(client=mock_client)

        with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
            await planner.plan("Create something")

    @pytest.mark.asyncio
    async def test_planner_includes_goal_in_prompt(self, mock_client):
        """Test planner includes goal in prompt."""
        response = json.dumps({
            "tasks": [{"id": "1", "description": "task", "files": [], "dependencies": []}],
            "rationale": "test",
        })
        mock_client.complete.return_value = response

        planner = PlannerAgent(client=mock_client)
        await planner.plan("Build a web scraper")

        # Verify complete was called
        mock_client.complete.assert_called_once()
        call_args = mock_client.complete.call_args

        # Check that the prompt contains the goal
        prompt = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        assert "Build a web scraper" in prompt

    @pytest.mark.asyncio
    async def test_planner_uses_planner_model(self, mock_client):
        """Test planner uses the planner_model setting."""
        response = json.dumps({
            "tasks": [{"id": "1", "description": "task", "files": [], "dependencies": []}],
            "rationale": "test",
        })
        mock_client.complete.return_value = response

        planner = PlannerAgent(client=mock_client)
        await planner.plan("Test goal")

        # Verify the correct model was used
        call_kwargs = mock_client.complete.call_args[1]
        assert call_kwargs.get("model") == "test-planner-model"

    @pytest.mark.asyncio
    async def test_planner_includes_planner_prompt(self, mock_client):
        """Test planner includes the PLANNER_PROMPT in the request."""
        response = json.dumps({
            "tasks": [{"id": "1", "description": "task", "files": [], "dependencies": []}],
            "rationale": "test",
        })
        mock_client.complete.return_value = response

        planner = PlannerAgent(client=mock_client)
        await planner.plan("Test goal")

        call_args = mock_client.complete.call_args
        prompt = call_args[0][0]
        assert PLANNER_PROMPT in prompt

    @pytest.mark.asyncio
    async def test_planner_raises_on_missing_required_fields(self, mock_client):
        """Test planner raises when JSON is missing required fields."""
        # Missing 'rationale' field
        response = json.dumps({
            "tasks": [{"id": "1", "description": "task"}],
        })
        mock_client.complete.return_value = response

        planner = PlannerAgent(client=mock_client)

        with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
            await planner.plan("Test goal")

    @pytest.mark.asyncio
    async def test_planner_handles_empty_tasks(self, mock_client):
        """Test planner handles empty tasks list."""
        response = json.dumps({
            "tasks": [],
            "rationale": "Nothing to do",
        })
        mock_client.complete.return_value = response

        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("Empty goal")

        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 0
