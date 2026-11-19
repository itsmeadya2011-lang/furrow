from __future__ import annotations

import asyncio

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, Settings


class FakeLLMClient:
    def __init__(self, response="", raises=None):
        self.response = response
        self.raises = raises
        self.prompts = []
        self.models = []
        self._queue: list[str] | None = None

    async def complete(self, prompt, system="", model=None):
        self.prompts.append(prompt)
        self.models.append(model)
        if self.raises:
            raise self.raises
        if self._queue is not None:
            return self._queue.pop(0)
        return self.response

    def set_responses(self, responses: list[str]) -> None:
        """Queue up responses to be returned in order on successive calls."""
        self._queue = list(responses)


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


def test_tasks_have_files_and_dependencies():
    client = FakeLLMClient(
        response=(
            '{"tasks":[{"id":"1","description":"write module",'
            '"files":["src/mod.py"],"dependencies":[]},'
            '{"id":"2","description":"write tests",'
            '"files":["tests/test_mod.py"],"dependencies":["1"]}],'
            '"rationale":"write module then tests"}'
        )
    )
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("goal"))
    assert len(plan.tasks) == 2
    assert plan.tasks[0].files == ["src/mod.py"]
    assert plan.tasks[0].dependencies == []
    assert plan.tasks[1].files == ["tests/test_mod.py"]
    assert plan.tasks[1].dependencies == ["1"]


def test_missing_tasks_defaults_to_empty():
    client = FakeLLMClient(response='{"rationale":"no tasks"}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert plan.tasks == []


def test_missing_rationale_defaults_to_empty():
    client = FakeLLMClient(response='{"tasks":[{"id":"1","description":"do thing"}]}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert plan.rationale == ""


def test_second_attempt_gets_stricter_prompt():
    """When the first parse fails, the planner retries with a stricter prompt."""
    client = FakeLLMClient()
    client.set_responses(["not valid json", '{"tasks":[],"rationale":"recovered"}'])
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert plan.tasks == []
    assert plan.rationale == "recovered"
    # The second prompt should contain the "could not be parsed" instruction
    assert len(client.prompts) == 2
    assert "could not be parsed" in client.prompts[1].lower()
    assert "no prose" in client.prompts[1].lower()
