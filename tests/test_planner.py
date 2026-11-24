from __future__ import annotations

import asyncio

from furrow.agents.planner import MAX_PLANNER_ATTEMPTS, PlannerAgent
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


class SequenceFakeLLMClient:
    """Returns a different canned response for each call to ``complete``."""

    def __init__(self, responses, raises=None):
        self.responses = list(responses)
        self.raises = raises
        self.prompts = []
        self.models = []

    async def complete(self, prompt, system="", model=None):
        self.prompts.append(prompt)
        self.models.append(model)
        if self.raises:
            raise self.raises
        if not self.responses:
            raise AssertionError("no more canned responses")
        return self.responses.pop(0)


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


def test_succeeds_after_repeated_failures():
    """Fails twice, succeeds on the third attempt; verifies result and call count."""
    responses = [
        "not json at all",
        "```json\n{broken```",
        '{"tasks":[{"id":"3","description":"recovered"}],"rationale":"recovered"}',
    ]
    client = SequenceFakeLLMClient(responses)
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert isinstance(plan, Plan)
    assert len(client.prompts) == 3
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "3"
    assert plan.rationale == "recovered"


def test_all_attempts_fail_returns_empty_plan():
    """When every attempt fails, plan() returns an empty plan instead of raising."""
    client = FakeLLMClient(response="not json at all")
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert isinstance(plan, Plan)
    assert plan.tasks == []
    assert plan.rationale == "Planner output unparseable; goal treated as complete."
    assert len(client.prompts) == MAX_PLANNER_ATTEMPTS


def test_retry_prompt_gets_stricter():
    """Each retry appends an increasingly strict instruction to the prompt."""
    client = FakeLLMClient(response="not json at all")
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    run(agent.plan("g"))
    assert len(client.prompts) == MAX_PLANNER_ATTEMPTS
    assert client.prompts[0] == client.prompts[1][: len(client.prompts[0])]
    for later, earlier in zip(client.prompts[1:], client.prompts[:-1]):
        assert len(later) > len(earlier)
        assert earlier == later[: len(earlier)]


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
