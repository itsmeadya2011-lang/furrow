from __future__ import annotations

import pytest

from furrow.agents.planner import PlannerAgent, extract_json
from furrow.config import Settings
from tests.conftest import FakeLLMClient, make_plan_json


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == '{"a": 1}'


def test_extract_json_fenced():
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_extract_json_with_prose():
    assert extract_json('Sure! Here it is:\n{"a": 1}\nHope this helps.') == '{"a": 1}'


def test_extract_json_no_json():
    with pytest.raises(ValueError):
        extract_json("no json here")


async def test_plan_parses_fenced_json_and_caps_tasks():
    settings = Settings(max_parallel_tasks=2)
    plan_payload = make_plan_json(
        tasks=[
            {"id": 1, "description": f"task {i}", "files": [], "dependencies": []}
            for i in range(4)
        ],
        rationale="four tasks",
    )
    fake = FakeLLMClient(plans=[plan_payload], settings=settings)
    plan = await PlannerAgent(client=fake).plan("g")
    assert len(plan.tasks) == 2
    assert all(isinstance(t.id, str) for t in plan.tasks)
    assert plan.rationale == "four tasks"