"""Tests for the planner agent's JSON parsing and retry logic."""
from __future__ import annotations

import json

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, Settings


class _FakeLLMClient:
    """A fake LLMClient that returns canned responses."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._call_count = 0
        self.settings = Settings()

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        if self._call_count >= len(self._responses):
            return self._responses[-1]
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp


@pytest.mark.asyncio
async def test_plan_valid_json():
    """Planner successfully parses valid JSON on first attempt."""
    plan_data = {
        "tasks": [{"id": "1", "description": "do thing", "files": [], "dependencies": []}],
        "rationale": "some rationale",
    }
    client = _FakeLLMClient(responses=[json.dumps(plan_data)])
    planner = PlannerAgent(client=client)
    plan = await planner.plan("some goal")
    assert isinstance(plan, Plan)
    assert plan.rationale == "some rationale"
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "do thing"


@pytest.mark.asyncio
async def test_plan_retries_on_invalid_json():
    """Planner retries when LLM returns invalid JSON, then succeeds."""
    plan_data = {
        "tasks": [{"id": "1", "description": "do thing", "files": [], "dependencies": []}],
        "rationale": "some rationale",
    }
    client = _FakeLLMClient(
        responses=["not json at all", "also not json", json.dumps(plan_data)]
    )
    planner = PlannerAgent(client=client)
    plan = await planner.plan("some goal")
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1


@pytest.mark.asyncio
async def test_plan_raises_after_max_retries():
    """Planner raises ValueError after exhausting retries."""
    client = _FakeLLMClient(responses=["not json", "not json", "not json"])
    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("some goal")


@pytest.mark.asyncio
async def test_plan_validates_required_keys():
    """Planner rejects JSON missing 'tasks' or 'rationale'."""
    client = _FakeLLMClient(
        responses=[json.dumps({"tasks": []}), json.dumps({"foo": "bar"}), json.dumps({})]
    )
    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("some goal")
