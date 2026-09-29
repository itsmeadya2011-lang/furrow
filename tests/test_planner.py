"""Unit tests for the PlannerAgent retry logic."""

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, Settings


class FakeLLMClient:
    """LLM client stub that returns canned responses in sequence."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._index = 0
        self._settings = Settings()
        self.calls = 0

    @property
    def settings(self):
        return self._settings

    async def complete(self, prompt, system="", model=None, timeout=None):
        self.calls += 1
        if self._index < len(self._responses):
            resp = self._responses[self._index]
            self._index += 1
            return resp
        return self._responses[-1]


@pytest.mark.asyncio
async def test_plan_success_on_first_try():
    """Planner should return a Plan when the LLM responds with valid JSON."""
    response = '{"tasks": [{"id": "1", "description": "build API"}], "rationale": "simple"}'
    client = FakeLLMClient([response])
    planner = PlannerAgent(client=client)

    plan = await planner.plan("Build an API")

    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "build API"
    assert plan.rationale == "simple"
    assert client.calls == 1


@pytest.mark.asyncio
async def test_plan_retries_on_invalid_json():
    """Planner should retry when LLM returns invalid JSON, up to max_retries."""
    bad_response = "not json at all"
    good_response = '{"tasks": [{"id": "1", "description": "fix bug"}], "rationale": "retry worked"}'
    client = FakeLLMClient([bad_response, good_response])
    planner = PlannerAgent(client=client)

    plan = await planner.plan("Fix a bug")

    assert plan.tasks[0].description == "fix bug"
    assert client.calls == 2


@pytest.mark.asyncio
async def test_plan_fails_after_max_retries():
    """Planner should raise ValueError after exhausting retries."""
    client = FakeLLMClient(["not json", "also not json", "still not json"])
    planner = PlannerAgent(client=client)

    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("Do something")

    # Initial attempt + 2 retries = 3 calls (max_retries=3)
    assert client.calls == 3


@pytest.mark.asyncio
async def test_plan_empty_tasks():
    """Planner should handle a response with no tasks."""
    response = '{"tasks": [], "rationale": "nothing to do"}'
    client = FakeLLMClient([response])
    planner = PlannerAgent(client=client)

    plan = await planner.plan("Solve everything")

    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 0
    assert plan.rationale == "nothing to do"
