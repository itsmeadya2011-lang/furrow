"""Tests for the PlannerAgent."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan


@pytest.mark.asyncio
async def test_planner_parse_valid_plan(mock_llm_client: MagicMock):
    """Planner returns a valid Plan when LLM output is valid JSON."""
    plan_json = json.dumps({
        "tasks": [
            {
                "id": "1",
                "description": "Add type hints",
                "files": ["src/utils.py"],
                "dependencies": [],
            },
            {
                "id": "2",
                "description": "Add tests for utils",
                "files": ["tests/test_utils.py"],
                "dependencies": ["1"],
            },
        ],
        "rationale": "Type safety and tests",
    })
    mock_llm_client.complete = AsyncMock(return_value=plan_json)

    planner = PlannerAgent(client=mock_llm_client)
    plan = await planner.plan("Add type hints and tests")

    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 2
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "Add type hints"
    assert plan.tasks[1].dependencies == ["1"]
    assert plan.rationale == "Type safety and tests"

    # Verify the correct model was used
    mock_llm_client.complete.assert_awaited_once()
    args, kwargs = mock_llm_client.complete.call_args
    assert kwargs["model"] == mock_llm_client.settings.planner_model


@pytest.mark.asyncio
async def test_planner_parse_error_raises(mock_llm_client: MagicMock):
    """Planner raises ValueError when LLM returns invalid JSON."""
    mock_llm_client.complete = AsyncMock(return_value="not valid json at all")

    planner = PlannerAgent(client=mock_llm_client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("Some goal")


@pytest.mark.asyncio
async def test_planner_uses_correct_model(mock_llm_client: MagicMock):
    """Planner passes its configured model to the client."""
    plan_json = json.dumps({"tasks": [], "rationale": "done"})
    mock_llm_client.complete = AsyncMock(return_value=plan_json)

    planner = PlannerAgent(client=mock_llm_client)
    await planner.plan("Goal")

    call_kwargs = mock_llm_client.complete.call_args.kwargs
    assert call_kwargs["model"] == mock_llm_client.settings.planner_model


@pytest.mark.asyncio
async def test_planner_empty_tasks_plan(mock_llm_client: MagicMock):
    """Planner can handle a plan with no tasks (goal already complete)."""
    plan_json = json.dumps({"tasks": [], "rationale": "Goal is already complete"})
    mock_llm_client.complete = AsyncMock(return_value=plan_json)

    planner = PlannerAgent(client=mock_llm_client)
    plan = await planner.plan("Check that everything is fine")

    assert plan.tasks == []
    assert "complete" in plan.rationale.lower()
