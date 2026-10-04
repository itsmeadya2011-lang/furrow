"""Agent-layer tests: JSON extraction, plan parsing, tester verdict, model routing.

Every agent gets a fake client, so nothing here reaches a provider. The tester
fixture additionally blocks subprocess creation so no real test runner is spawned.
"""

import asyncio
import json

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.prompts import extract_json_object
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel

TIMEOUT = 10

PASSED_JSON = '{"passed": true, "summary": "all green", "failures": []}'
FAILED_JSON = '```json\n{"passed": false, "summary": "1 failed", "failures": ["boom"]}\n```'
PLAN_DICT = {
    "tasks": [{"id": "1", "description": "Add a widget factory", "files": ["widgets.py"]}],
    "rationale": "one slice is enough",
}
FENCED_PLAN = "```json\n" + json.dumps(PLAN_DICT) + "\n```"


class FakeLLMClient:
    """Stand-in for ``LLMClient``: records calls and replays canned text."""

    def __init__(self, responses=(), **settings_overrides):
        self.responses = list(responses)
        self.calls = []
        self.settings = Settings(_env_file=None, **settings_overrides)

    async def complete(self, prompt, system="", model=None):
        self.calls.append((prompt, model))
        if self.responses:
            return self.responses.pop(0)
        return "{}"


@pytest.fixture
def no_subprocesses(monkeypatch):
    """Guarantee the tester never spawns a real test runner inside the test run."""
    async def _no_runner(*_args, **_kwargs):
        raise FileNotFoundError("no test runner available in unit tests")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _no_runner)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", _no_runner)
    monkeypatch.setattr(asyncio.subprocess, "create_subprocess_exec", _no_runner)
    monkeypatch.setattr(asyncio.subprocess, "create_subprocess_shell", _no_runner)

    async def _fake_run_tests(self):
        return "1 passed, 0 failed"

    monkeypatch.setattr(TesterAgent, "_run_tests", _fake_run_tests, raising=False)
    return monkeypatch


# --------------------------------------------------------------------------- #
# extract_json_object
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param('{"a": 1}', {"a": 1}, id="bare"),
        pytest.param('```json\n{"a": 1}\n```', {"a": 1}, id="json-fence"),
        pytest.param('```\n{"a": 1}\n```', {"a": 1}, id="plain-fence"),
        pytest.param(
            'Here you go: {"a": 1} hope that helps', {"a": 1}, id="surrounding-prose"
        ),
        pytest.param('{"a": "}{"}', {"a": "}{"}, id="braces-inside-string"),
    ],
)
def test_extract_json_object_parses_tolerantly(raw, expected):
    assert extract_json_object(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("", id="empty"),
        pytest.param("not json at all", id="prose-only"),
        pytest.param('{"a":', id="truncated"),
    ],
)
def test_extract_json_object_raises_value_error(raw):
    with pytest.raises(ValueError):
        extract_json_object(raw)


# --------------------------------------------------------------------------- #
# PlannerAgent
# --------------------------------------------------------------------------- #


async def test_planner_parses_fenced_json():
    client = FakeLLMClient([FENCED_PLAN])
    planner = PlannerAgent(client=client, settings=client.settings)

    plan = await asyncio.wait_for(planner.plan("ship it"), timeout=TIMEOUT)

    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "Add a widget factory"
    assert plan.tasks[0].id == "1"
    assert plan.rationale == "one slice is enough"


async def test_planner_parses_bare_json():
    client = FakeLLMClient([json.dumps(PLAN_DICT)])
    planner = PlannerAgent(client=client, settings=client.settings)

    plan = await asyncio.wait_for(planner.plan("ship it"), timeout=TIMEOUT)

    assert plan.tasks[0].description == "Add a widget factory"


async def test_planner_raises_value_error_on_garbage():
    client = FakeLLMClient(["I am not going to emit JSON today."])
    planner = PlannerAgent(client=client, settings=client.settings)

    with pytest.raises(ValueError):
        await asyncio.wait_for(planner.plan("ship it"), timeout=TIMEOUT)


async def test_planner_raises_on_missing_tasks_key():
    client = FakeLLMClient(['{"rationale": "x"}'])
    planner = PlannerAgent(client=client, settings=client.settings)

    with pytest.raises(ValueError):
        await asyncio.wait_for(planner.plan("ship it"), timeout=TIMEOUT)


async def test_planner_uses_planner_model():
    client = FakeLLMClient([json.dumps(PLAN_DICT)])
    planner = PlannerAgent(client=client, settings=client.settings)

    await asyncio.wait_for(planner.plan("ship it"), timeout=TIMEOUT)

    assert len(client.calls) == 1
    prompt, model = client.calls[0]
    assert model == client.settings.planner_model
    assert "ship it" in prompt


# --------------------------------------------------------------------------- #
# TesterAgent
# --------------------------------------------------------------------------- #


async def test_tester_unparseable_response_is_not_a_pass(no_subprocesses):
    # Regression: "passed" in response.lower() used to turn this into a false PASS.
    client = FakeLLMClient(["All tests did not pass"])
    tester = TesterAgent(client=client, settings=client.settings)

    result = await asyncio.wait_for(tester.run("ship it", []), timeout=TIMEOUT)

    assert result.passed is False


async def test_tester_parses_fenced_json(no_subprocesses):
    client = FakeLLMClient([FAILED_JSON])
    tester = TesterAgent(client=client, settings=client.settings)

    result = await asyncio.wait_for(tester.run("ship it", []), timeout=TIMEOUT)

    assert result.passed is False
    assert result.failures == ["boom"]
    assert result.summary == "1 failed"


async def test_tester_parses_passing_json(no_subprocesses):
    client = FakeLLMClient([PASSED_JSON])
    tester = TesterAgent(client=client, settings=client.settings)

    result = await asyncio.wait_for(tester.run("ship it", []), timeout=TIMEOUT)

    assert result.passed is True
    assert result.failures == []


async def test_tester_uses_tester_model(no_subprocesses):
    client = FakeLLMClient([PASSED_JSON])
    tester = TesterAgent(client=client, settings=client.settings)

    await asyncio.wait_for(tester.run("ship it", []), timeout=TIMEOUT)

    assert len(client.calls) == 1
    prompt, model = client.calls[0]
    assert model == client.settings.tester_model
    assert "ship it" in prompt


# --------------------------------------------------------------------------- #
# WorkerAgent
# --------------------------------------------------------------------------- #


async def test_worker_uses_worker_model():
    task = TaskModel(
        id="1",
        description="Add a widget factory",
        files=["furrow/widgets.py", "tests/test_widgets.py"],
    )
    client = FakeLLMClient(["Added the factory and a test."])
    worker = WorkerAgent(task=task, client=client, settings=client.settings)

    summary = await asyncio.wait_for(worker.run(), timeout=TIMEOUT)

    assert summary == "Added the factory and a test."
    assert len(client.calls) == 1
    prompt, model = client.calls[0]
    assert model == client.settings.worker_model
    assert "Add a widget factory" in prompt
    assert "furrow/widgets.py" in prompt
    assert "tests/test_widgets.py" in prompt