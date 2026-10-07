from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from furrow.agents.planner import PlannerAgent, _extract_json
from furrow.config import Settings


def test_extract_json_plain():
    data = '{"key": "value"}'
    assert _extract_json(data) == data


def test_extract_json_fenced():
    response = "```json\n{\"key\": \"value\"}\n```"
    assert _extract_json(response) == '{"key": "value"}'


async def test_plan_parses_fenced_response():
    client = AsyncMock()
    client.settings = Settings(_env_file=None)
    client.complete = AsyncMock(
        return_value=(
            '```json\n{"tasks": [{"id": "1", "description": "d", '
            '"files": [], "dependencies": []}], "rationale": "r"}\n```'
        )
    )
    planner = PlannerAgent(client=client)
    plan = await planner.plan("test")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"


async def test_plan_retries_then_succeeds():
    client = AsyncMock()
    client.settings = Settings(_env_file=None)
    client.complete = AsyncMock(
        side_effect=[
            "invalid json",
            (
                '{"tasks": [{"id": "1", "description": "d", '
                '"files": [], "dependencies": []}], "rationale": "r"}'
            ),
        ]
    )
    planner = PlannerAgent(client=client)
    plan = await planner.plan("test")
    assert len(plan.tasks) == 1
    assert client.complete.call_count == 2


async def test_plan_raises_after_three_failures():
    client = AsyncMock()
    client.settings = Settings(_env_file=None)
    client.complete = AsyncMock(return_value="invalid json")
    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("test")
    assert client.complete.call_count == 3
