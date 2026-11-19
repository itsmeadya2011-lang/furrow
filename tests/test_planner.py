from __future__ import annotations

import asyncio

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, Settings
from furrow.config import settings as global_settings


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


class SequencedLLMClient:
    """Fake client that returns (or raises) a different value on each call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.models = []
        self.calls = 0

    async def complete(self, prompt, system="", model=None):
        self.prompts.append(prompt)
        self.models.append(model)
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        item = self.responses[index]
        if isinstance(item, BaseException):
            raise item
        return item


class FlakyLLMClient:
    """Raises on the first call, succeeds on every later call."""

    def __init__(self, error, response):
        self.error = error
        self.response = response
        self.prompts = []
        self.models = []
        self.calls = 0

    async def complete(self, prompt, system="", model=None):
        self.prompts.append(prompt)
        self.models.append(model)
        self.calls += 1
        if self.calls == 1:
            raise self.error
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


def test_retry_first_parse_fails_second_succeeds():
    client = SequencedLLMClient(
        [
            "this is not json",
            '{"tasks":[{"id":"3","description":"recovered"}],"rationale":"ok"}',
        ]
    )
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert client.calls == 2
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "3"
    assert plan.rationale == "ok"
    assert "could not be parsed" in client.prompts[1]
    assert "ONLY a JSON object" in client.prompts[1]


def test_retry_both_attempts_fail_raises_value_error():
    client = SequencedLLMClient(["nope", "still nope"])
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    with pytest.raises(ValueError) as excinfo:
        run(agent.plan("g"))
    message = str(excinfo.value)
    assert client.calls == 2
    assert "attempt 1" in message
    assert "attempt 2" in message
    assert "still nope" in message
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_empty_goal_still_builds_prompt():
    client = FakeLLMClient(response='{"tasks":[],"rationale":""}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan(""))
    assert plan.tasks == []
    assert "Goal: " in client.prompts[0]
    assert client.prompts[0].rstrip().endswith("Goal:")


def test_goal_with_special_characters_is_preserved():
    goal = 'Fix "quotes", {braces}, $vars, 100% & <tags>\nsecond line — ünïcode'
    client = FakeLLMClient(response='{"tasks":[],"rationale":""}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    run(agent.plan(goal))
    assert goal in client.prompts[0]


def test_large_context_with_test_failures_is_included():
    failures = "\n".join(
        f"FAILED tests/test_mod_{i}.py::test_case_{i} - AssertionError: expected {i}"
        for i in range(80)
    )
    context = "Test output:\n" + failures
    client = FakeLLMClient(response='{"tasks":[],"rationale":""}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    run(agent.plan("fix the failing tests", context=context))
    prompt = client.prompts[0]
    assert context in prompt
    assert "FAILED tests/test_mod_79.py" in prompt
    assert "Context:" in prompt


def test_default_settings_uses_global_settings():
    client = FakeLLMClient(response='{"tasks":[],"rationale":""}')
    agent = PlannerAgent(client=client)
    assert agent.settings is global_settings
    run(agent.plan("g"))
    assert client.models[0] == global_settings.planner_model


def test_custom_client_raising_first_call_propagates():
    client = FlakyLLMClient(
        error=RuntimeError("transient failure"),
        response='{"tasks":[],"rationale":""}',
    )
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    with pytest.raises(RuntimeError, match="transient failure"):
        run(agent.plan("g"))
    assert client.calls == 1


def test_custom_client_raising_value_error_first_call_propagates():
    client = FlakyLLMClient(
        error=ValueError("boom"),
        response='{"tasks":[],"rationale":""}',
    )
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    with pytest.raises(ValueError, match="boom"):
        run(agent.plan("g"))
    assert client.calls == 1


def test_plan_with_multiple_tasks_and_dependencies():
    client = FakeLLMClient(
        response="""{
            "tasks": [
                {"id": "1", "description": "setup", "files": ["a.py"], "dependencies": []},
                {"id": "2", "description": "core", "files": ["b.py"], "dependencies": ["1"]},
                {"id": "3", "description": "tests", "dependencies": ["1", "2"]}
            ],
            "rationale": "layered plan"
        }"""
    )
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("build it"))
    assert len(plan.tasks) == 3
    assert [t.id for t in plan.tasks] == ["1", "2", "3"]
    assert plan.tasks[0].dependencies == []
    assert plan.tasks[1].dependencies == ["1"]
    assert plan.tasks[2].dependencies == ["1", "2"]
    assert plan.tasks[1].files == ["b.py"]
    assert plan.rationale == "layered plan"
    assert all(t.status == "pending" for t in plan.tasks)


def test_missing_rationale_defaults_to_empty_string():
    client = FakeLLMClient(response='{"tasks":[{"id":"1","description":"d"}]}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert plan.rationale == ""
    assert len(plan.tasks) == 1


def test_missing_tasks_defaults_to_empty_list():
    client = FakeLLMClient(response='{"rationale":"nothing to do"}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert plan.tasks == []
    assert plan.rationale == "nothing to do"
