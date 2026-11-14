from __future__ import annotations

import asyncio

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, Settings


class FakeLLMClient:
    def __init__(self, response="", raises=None):
        self.response = response
        self.raises = raises
        self.prompts = []
        self.models = []

    async def complete(self, prompt, system="", model=None):
        self.prompts.append(prompt)
        self.models.append(model)
        if self.raises:
            raise self.raises
        return self.response


def run(coro):
    """Run a coroutine to completion on a fresh event loop."""
    return asyncio.run(coro)


def test_valid_json():
    client = FakeLLMClient(
        response='{"tasks":[{"id":"1","description":"do thing"}],"rationale":"r"}'
    )
    s = Settings(_env_file=None)
    agent = PlannerAgent(client=client, settings=s)
    plan = run(agent.plan("goal"))
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"


def test_fenced_json():
    client = FakeLLMClient(response='```json\n{"tasks":[],"rationale":"x"}\n```')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert plan.tasks == []


def test_prose_surrounded():
    client = FakeLLMClient(
        response='Here is plan: {"tasks":[{"id":"2","description":"d"}],"rationale":"y"} thanks'
    )
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert plan.tasks[0].id == "2"


def test_invalid_twice_raises():
    client = FakeLLMClient(response="not json at all")
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    try:
        run(agent.plan("g"))
        raised = False
    except ValueError as e:
        raised = True
        assert "not json" in str(e) or "response" in str(e).lower() or len(str(e)) > 0
    assert raised


def test_context_included():
    client = FakeLLMClient(response='{"tasks":[],"rationale":""}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    run(agent.plan("g", context="FAILING TESTS"))
    assert "FAILING TESTS" in client.prompts[0]


def test_uses_planner_model():
    client = FakeLLMClient(response='{"tasks":[],"rationale":""}')
    s = Settings(_env_file=None, planner_model="my-model")
    agent = PlannerAgent(client=client, settings=s)
    run(agent.plan("g"))
    assert client.models[0] == "my-model"


class TwoShotClient:
    """Fake LLM client that returns a different response on each call."""

    def __init__(self, first: str, second: str):
        self.first = first
        self.second = second
        self.prompts: list[str] = []
        self.models: list[str | None] = []

    async def complete(self, prompt, system="", model=None):
        self.prompts.append(prompt)
        self.models.append(model)
        if len(self.prompts) == 1:
            return self.first
        return self.second


def test_stricter_retry_recovers_when_second_attempt_is_valid():
    """The planner retries once with stricter instructions on parse failure.

    When the first response is unparseable and the second (stricter) response
    is valid JSON, the planner returns the second plan instead of raising.
    """
    client = TwoShotClient(
        first="not json at all",
        second='{"tasks":[{"id":"1","description":"do thing"}],"rationale":"recovered"}',
    )
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.rationale == "recovered"
    # Two completions were issued: the original prompt and the stricter one.
    assert len(client.prompts) == 2
    assert "could not be parsed" in client.prompts[1]


def test_stricter_retry_still_raises_when_both_attempts_are_invalid():
    """Two invalid responses exhaust the retry budget and raise."""
    client = TwoShotClient(first="not json at all", second="still not json")
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    with pytest.raises(ValueError):
        run(agent.plan("g"))
    assert len(client.prompts) == 2
