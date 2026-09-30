import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class FakeClient:
    """Minimal LLMClient stand-in that avoids constructing real provider clients."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_config_new_defaults():
    s = Settings()
    assert s.max_retries >= 1
    assert s.ollama_model == "llama3"
    assert s.max_parallel_tasks == 5


def test_is_done_no_plan():
    o = Orchestrator(goal="x", client=FakeClient())
    assert o._is_done() is True


def test_is_done_all_completed_tests_pass():
    o = Orchestrator(goal="x", client=FakeClient())
    t = TaskModel(id="1", description="d")
    o.tasks = [t]
    t.status = "completed"
    o.last_plan = Plan(tasks=[t], rationale="r")
    o.last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert o._is_done() is True


def test_is_done_tests_failing_not_done():
    o = Orchestrator(goal="x", client=FakeClient())
    t = TaskModel(id="1", description="d")
    o.tasks = [t]
    t.status = "completed"
    o.last_plan = Plan(tasks=[t], rationale="r")
    o.last_test_result = TestResult(passed=False, summary="bad", failures=["bad"])
    assert o._is_done() is False


def test_is_done_task_failed_not_done():
    o = Orchestrator(goal="x", client=FakeClient())
    t = TaskModel(id="1", description="d")
    o.tasks = [t]
    t.status = "failed"
    t.result = "err"
    o.last_plan = Plan(tasks=[t], rationale="r")
    o.last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert o._is_done() is False


def test_max_cycles_respected():
    o = Orchestrator(goal="x", client=FakeClient())
    o.settings = Settings(max_cycles=1)
    o.cycles = 1
    assert o._max_cycles_reached() is True

_FAKE_PLAN = Plan(tasks=[TaskModel(id="1", description="d")], rationale="r")
_FAKE_TEST_PASS = TestResult(passed=True, summary="ok", failures=[])


async def _fake_plan(self, goal: str) -> Plan:
    return _FAKE_PLAN


async def _fake_worker(self) -> str:
    return "done"


async def _fake_tester_pass(self, goal: str, tasks: list) -> TestResult:
    return _FAKE_TEST_PASS


@pytest.mark.asyncio
async def test_run_terminates_on_max_cycles(monkeypatch):
    o = Orchestrator(goal="demo", client=FakeClient())
    o.settings = Settings(max_cycles=1)
    monkeypatch.setattr(PlannerAgent, "plan", _fake_plan)
    monkeypatch.setattr(WorkerAgent, "run", _fake_worker)
    monkeypatch.setattr(TesterAgent, "run", _fake_tester_pass)
    await o.run()
    assert o.cycles == 1
    assert o.tasks[0].status == "completed"
    assert o.last_test_result.passed is True


@pytest.mark.asyncio
async def test_run_continues_when_tests_fail_then_passes(monkeypatch):
    o = Orchestrator(goal="demo", client=FakeClient())
    o.settings = Settings(max_cycles=5)
    state = {"fails": 0}

    async def plan(self, goal: str) -> Plan:
        return _FAKE_PLAN

    async def worker(self) -> str:
        return "done"

    async def tester(self, goal: str, tasks: list) -> TestResult:
        state["fails"] += 1
        if state["fails"] == 1:
            return TestResult(passed=False, summary="bad", failures=["bad test"])
        return _FAKE_TEST_PASS

    monkeypatch.setattr(PlannerAgent, "plan", plan)
    monkeypatch.setattr(WorkerAgent, "run", worker)
    monkeypatch.setattr(TesterAgent, "run", tester)
    await o.run()
    assert o.cycles == 2
    assert o.last_test_result.passed is True


@pytest.mark.asyncio
async def test_complete_retries_then_succeeds(monkeypatch):
    c = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="x"))
    calls = {"n": 0}

    async def flaky(prompt: str, system: str, model: str) -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient failure")
        return "ok"

    monkeypatch.setattr(c, "_complete_anthropic", flaky)
    result = await c.complete("hi")
    assert result == "ok"
    assert calls["n"] == 2
