"""Tests for the PlannerAgent."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.settings.planner_model = "claude-3-5-haiku-20241022"
    client.complete = AsyncMock()
    return client


class TestPlanParse:
    @pytest.mark.asyncio
    async def test_valid_plan(self, mock_client):
        plan_json = json.dumps({
            "tasks": [
                {"id": "1", "description": "create auth module", "files": ["auth.py"], "dependencies": []},
                {"id": "2", "description": "add tests for auth", "files": ["test_auth.py"], "dependencies": ["1"]},
            ],
            "rationale": "Break auth into implementation and tests",
        })
        mock_client.complete.return_value = plan_json

        agent = PlannerAgent(client=mock_client)
        plan = await agent.plan("Add JWT authentication")

        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 2
        assert plan.tasks[0].id == "1"
        assert plan.tasks[0].description == "create auth module"
        assert plan.tasks[0].files == ["auth.py"]
        assert plan.tasks[1].dependencies == ["1"]
        assert "auth" in plan.rationale.lower() or "break" in plan.rationale.lower()

    @pytest.mark.asyncio
    async def test_empty_plan(self, mock_client):
        plan_json = json.dumps({"tasks": [], "rationale": "Goal is simple, no tasks needed"})
        mock_client.complete.return_value = plan_json

        agent = PlannerAgent(client=mock_client)
        plan = await agent.plan("Do nothing")

        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 0

    @pytest.mark.asyncio
    async def test_invalid_json_raises(self, mock_client):
        mock_client.complete.return_value = "This is not JSON"

        agent = PlannerAgent(client=mock_client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await agent.plan("test")

    @pytest.mark.asyncio
    async def test_malformed_json_raises(self, mock_client):
        mock_client.complete.return_value = '{"tasks": "not a list"}'

        agent = PlannerAgent(client=mock_client)
        with pytest.raises((ValueError, Exception)):
            await agent.plan("test")

    @pytest.mark.asyncio
    async def test_passes_system_prompt(self, mock_client):
        mock_client.complete.return_value = json.dumps({"tasks": [], "rationale": "ok"})

        agent = PlannerAgent(client=mock_client)
        await agent.plan("test goal")

        args, kwargs = mock_client.complete.call_args
        prompt = args[0] if args else kwargs.get("prompt", "")
        assert "Goal: test goal" in prompt
