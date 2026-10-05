import json
from pathlib import Path

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


@pytest.fixture(autouse=True)
def stub_test_runner(monkeypatch):
    """Stop TesterAgent from shelling out to real test runners.

    ``TesterAgent.run()`` calls ``_run_tests()`` first, which spawns ``pytest -q``
    in the repo root. That subprocess re-collects this suite, whose tester tests
    spawn another one, and the recursion never terminates.
    """

    async def fake_run_tests(self) -> str:
        return "collected 0 items"

    monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests)


class FakeLLMClient(LLMClient):
    """LLMClient stub that returns canned responses without network calls."""

    def __init__(self, settings: Settings, responses: list[str]) -> None:
        # Bypass LLMClient.__init__ to avoid touching real settings singleton
        self.settings = settings
        self._anthropic = None
        self._openai = None
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("unexpected LLM call")
        return self._responses.pop(0)


def make_settings(**overrides) -> Settings:
    defaults = {
        "provider": "anthropic",
        "anthropic_api_key": "test-key",
        "max_cycles": 0,
        "state_file": None,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def plan_response(tasks: list[dict], rationale: str = "ok") -> str:
    return json.dumps({"tasks": tasks, "rationale": rationale})


def test_plan_parses_fenced_json():
    settings = make_settings()
    client = FakeLLMClient(
        settings, ["```json\n" + plan_response([{"id": "1", "description": "x"}]) + "\n```"]
    )
    planner = PlannerAgent(client=client)
    plan = asyncio_run(planner.plan("goal"))
    assert plan.tasks[0].id == "1"


def test_tester_parses_fenced_json():
    settings = make_settings()
    client = FakeLLMClient(
        settings, ['```json\n{"passed": true, "summary": "ok", "failures": []}\n```']
    )
    tester = TesterAgent(client=client)
    result = asyncio_run(tester.run("goal", []))
    assert result.passed is True


def test_tester_falls_back_on_unparseable_json():
    settings = make_settings()
    client = FakeLLMClient(settings, ["all tests passed hooray"])
    tester = TesterAgent(client=client)
    result = asyncio_run(tester.run("goal", []))
    assert result.passed is True
    assert "passed" in result.summary


def test_worker_returns_llm_result():
    settings = make_settings()
    client = FakeLLMClient(settings, ["did the thing"])
    task = TaskModel(id="1", description="do thing")
    worker = WorkerAgent(task=task, client=client)
    result = asyncio_run(worker.run())
    assert result == "did the thing"


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


def test_orchestrator_single_cycle_completion():
    settings = make_settings()
    responses = [
        plan_response([{"id": "1", "description": "do thing"}]),
        "did the thing",
        json.dumps({"passed": True, "summary": "ok", "failures": []}),
    ]
    client = FakeLLMClient(settings, responses)
    events: list[dict] = []
    orch = Orchestrator(goal="build a thing", client=client, on_event=events.append)
    asyncio_run(orch.run())

    assert orch.cycles == 1
    assert orch.goal == "build a thing"
    assert all(t.status == "completed" for t in orch.tasks)
    assert orch.last_test_passed is True
    assert orch.cycle_history[0]["cycle"] == 1
    assert orch.cycle_history[0]["test_passed"] is True

    types = [e["type"] for e in events]
    assert types[0] == "cycle.start"
    assert "plan" in types
    assert "task.start" in types
    assert "task.completed" in types
    assert "test.result" in types
    assert types[-1] == "done"


def test_orchestrator_fix_cycle_on_test_failure():
    settings = make_settings()
    responses = [
        plan_response([{"id": "1", "description": "do thing"}]),
        "did the thing",
        json.dumps({"passed": False, "summary": "boom", "failures": ["test_x failed"]}),
        # second cycle: planner receives the fix objective
        plan_response([{"id": "2", "description": "fix it"}]),
        "fixed it",
        json.dumps({"passed": True, "summary": "ok", "failures": []}),
    ]
    client = FakeLLMClient(settings, responses)
    orch = Orchestrator(goal="build a thing", client=client)
    asyncio_run(orch.run())

    assert orch.cycles == 2
    # Original goal must be preserved even after failure
    assert orch.goal == "build a thing"
    # The planner's second prompt must contain the fix objective
    assert any("Fix failing tests" in p for p in client.prompts)
    assert any("test_x failed" in p for p in client.prompts)


def test_orchestrator_halts_on_max_cycles():
    settings = make_settings(max_cycles=1)
    responses = [
        plan_response([{"id": "1", "description": "do thing"}]),
        "did the thing",
        json.dumps({"passed": False, "summary": "boom", "failures": ["f1"]}),
    ]
    client = FakeLLMClient(settings, responses)
    events: list[dict] = []
    orch = Orchestrator(goal="build a thing", client=client, on_event=events.append)
    asyncio_run(orch.run())

    assert orch.cycles == 1
    halted = [e for e in events if e["type"] == "halted"]
    assert halted and halted[0]["reason"] == "max_cycles"
    assert "done" not in [e["type"] for e in events]


def test_orchestrator_empty_plan_is_done():
    settings = make_settings()
    client = FakeLLMClient(settings, [plan_response([])])
    events: list[dict] = []
    orch = Orchestrator(goal="nothing to do", client=client, on_event=events.append)
    asyncio_run(orch.run())
    assert orch.cycles == 1
    assert [e["type"] for e in events][-1] == "done"


def test_orchestrator_failed_task_not_done():
    # max_cycles=1 so run() terminates: a failed task keeps _is_done() False, so
    # with the default max_cycles=0 (infinite) the loop would never exit.
    settings = make_settings(max_cycles=1)
    client = FakeLLMClient(
        settings,
        [
            plan_response([{"id": "1", "description": "do thing"}]),
            json.dumps({"passed": True, "summary": "ok", "failures": []}),
        ],
    )

    # Simulate a worker exception: monkeypatch WorkerAgent.run to raise
    async def raising_run(self):
        raise RuntimeError("worker exploded")

    import furrow.core.orchestrator as orch_mod

    orig = orch_mod.WorkerAgent.run
    orch_mod.WorkerAgent.run = raising_run
    events: list[dict] = []
    try:
        orch = Orchestrator(goal="build a thing", client=client, on_event=events.append)
        asyncio_run(orch.run())
    finally:
        orch_mod.WorkerAgent.run = orig

    assert orch.tasks[0].status == "failed"
    assert orch.last_test_passed is True
    # Tests passed but a task failed, so the run is not done; it halts on max_cycles.
    assert orch._is_done() is False
    types = [e["type"] for e in events]
    assert types.count("task.failed") == 1
    assert "task.completed" not in types
    assert types[-1] == "halted"
    assert events[-1]["reason"] == "max_cycles"


def test_orchestrator_persists_state_file(tmp_path: Path):
    state_file = tmp_path / "state" / "furrow-state.json"
    settings = make_settings(state_file=state_file)
    responses = [
        plan_response([{"id": "1", "description": "do thing"}]),
        "did the thing",
        json.dumps({"passed": True, "summary": "ok", "failures": []}),
    ]
    client = FakeLLMClient(settings, responses)
    orch = Orchestrator(goal="build a thing", client=client)
    asyncio_run(orch.run())

    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert data["goal"] == "build a thing"
    assert data["cycles"] == 1
    assert data["history"][0]["test_passed"] is True


def test_settings_state_file_default_is_none():
    s = Settings()
    assert s.state_file is None
