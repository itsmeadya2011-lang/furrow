from __future__ import annotations

import asyncio

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, Settings


class FakeLLMClient:
    def __init__(self, response="", responses=None, raises=None):
        self.responses = responses if responses is not None else [response]
        self.raises = raises
        self.prompts = []
        self.models = []
        self._call_count = 0

    async def complete(self, prompt, system="", model=None):
        self.prompts.append(prompt)
        self.models.append(model)
        if self.raises:
            raise self.raises
        if self._call_count < len(self.responses):
            resp = self.responses[self._call_count]
        else:
            resp = self.responses[-1]
        self._call_count += 1
        return resp


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


def test_retry_succeeds_on_second_attempt():
    client = FakeLLMClient(
        responses=['not json at all', '{"tasks":[{"id":"1","description":"d"}],"rationale":"r"}']
    )
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("goal"))
    assert len(client.prompts) == 2
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"


def test_both_attempts_fail_raises():
    client = FakeLLMClient(responses=["not json", "still not json"])
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    try:
        run(agent.plan("g"))
        raised = False
    except ValueError as e:
        raised = True
        assert "attempt 1" in str(e) and "attempt 2" in str(e)
    assert raised


def test_missing_rationale_defaults_empty():
    client = FakeLLMClient(response='{"tasks":[{"id":"1","description":"d"}]}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert plan.rationale == ""


def test_missing_tasks_defaults_empty():
    client = FakeLLMClient(response='{"rationale":"some rationale"}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert plan.tasks == []


def test_goal_included_in_prompt():
    client = FakeLLMClient(response='{"tasks":[],"rationale":""}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    run(agent.plan("my specific goal"))
    assert "my specific goal" in client.prompts[0]
