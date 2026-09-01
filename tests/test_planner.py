from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, TaskModel


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_parse(self) -> None:
        """When LLM returns valid plan JSON, Plan is parsed correctly."""
        mock_client = AsyncMock()
        mock_client.settings.planner_model = "claude-3-5-haiku-20241022"
        mock_client.complete.return_value = (
            '{"tasks": [{"id": "1", "description": "Do something", "files": [], "dependencies": []}], "rationale": "Simple plan"}'
        )

        agent = PlannerAgent(client=mock_client)
        result = await agent.plan("test goal")

        assert isinstance(result, Plan)
        assert len(result.tasks) == 1
        assert result.tasks[0].description == "Do something"
        assert result.rationale == "Simple plan"

    @pytest.mark.asyncio
    async def test_plan_parse_error(self) -> None:
        """When LLM returns invalid JSON, raises ValueError."""
        mock_client = AsyncMock()
        mock_client.settings.planner_model = "claude-3-5-haiku-20241022"
        mock_client.complete.return_value = "This is not valid JSON at all!"

        agent = PlannerAgent(client=mock_client)

        with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
            await agent.plan("test goal")
