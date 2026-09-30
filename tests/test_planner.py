"""Tests for furrow.agents.planner — JSON parsing and retry on invalid output."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan
from furrow.llm import LLMClient


def _make_planner():
    return PlannerAgent(client=LLMClient())


async def test_planner_parses_valid_json():
    planner = _make_planner()
    fake_response = json.dumps({
        "tasks": [
            {"id": "1", "description": "Add auth", "files": ["auth.py"], "dependencies": []},
        ],
        "rationale": "Need authentication",
    })
    with patch.object(planner.client, "complete", new=AsyncMock(return_value=fake_response)):
        plan = await planner.plan("Add JWT auth")
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "Add auth"
    assert plan.rationale == "Need authentication"


async def test_planner_retries_on_invalid_json():
    """Planner should retry when LLM returns invalid JSON."""
    planner = _make_planner()
    responses = [
        "not json at all",
        "also not json",
        json.dumps({
            "tasks": [{"id": "1", "description": "Fix bug", "files": [], "dependencies": []}],
            "rationale": "Fixing a bug",
        }),
    ]
    call_count = 0

    async def mock_complete(*args, **kwargs):
        nonlocal call_count
        result = responses[call_count]
        call_count += 1
        return result

    with patch.object(planner.client, "complete", new=mock_complete):
        plan = await planner.plan("Fix a bug")
    assert isinstance(plan, Plan)
    assert plan.tasks[0].description == "Fix bug"
    assert call_count == 3  # Tried 3 times (max retries = 3 attempts)


async def test_planner_retries_exhausted_raises_value_error():
    """If all retry attempts return invalid JSON, ValueError is raised."""
    planner = _make_planner()
    with (
        patch.object(planner.client, "complete", new=AsyncMock(return_value="invalid")),
        pytest.raises(ValueError, match="Failed to parse plan"),
    ):
        await planner.plan("Do something")


async def test_planner_parses_multiple_tasks():
    planner = _make_planner()
    fake_response = json.dumps({
        "tasks": [
            {"id": "1", "description": "Task A", "files": ["a.py"], "dependencies": []},
            {"id": "2", "description": "Task B", "files": ["b.py"], "dependencies": ["1"]},
            {"id": "3", "description": "Task C", "files": ["c.py"], "dependencies": ["1"]},
        ],
        "rationale": "Multi-step plan",
    })
    with patch.object(planner.client, "complete", new=AsyncMock(return_value=fake_response)):
        plan = await planner.plan("Big project")
    assert len(plan.tasks) == 3
    assert plan.tasks[1].dependencies == ["1"]
