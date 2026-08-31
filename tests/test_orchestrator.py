from __future__ import annotations

from types import SimpleNamespace

from furrow.config import Plan, TaskModel, TestResult, settings as global_settings
from furrow.core.orchestrator import Orchestrator

FAKE_SETTINGS = SimpleNamespace(planner_model="p", worker_model="w", tester_model="t")


class FakeClient:
    def __init__(self) -> None:
        self.settings = FAKE_SETTINGS

    async def complete(self, prompt: str, system: str = "", model=None) -> str:
        return "ok"

    async def write_file(self, path: str, content: str) -> None:
        pass


class FakePlanner:
    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    async def plan(self, goal: str) -> Plan:
        return self._plan


class FakeTester:
    def __init__(self, result: TestResult) -> None:
        self._result = result

    async def run(self, goal: str, tasks) -> TestResult:
        return self._result


def _make_orchestrator(goal: str = "g") -> Orchestrator:
    return Orchestrator(goal=goal, client=FakeClient())


def test_is_done_empty():
    o = _make_orchestrator()
    o.tasks = []
    assert o._is_done() is True


def test_is_done_pending():
    o = _make_orchestrator()
    o.tasks = [TaskModel(id="1", description="x")]
    assert o._is_done() is False


def test_is_done_all_completed():
    o = _make_orchestrator()
    o.tasks = [TaskModel(id="1", description="x", status="completed")]
    assert o._is_done() is True


def test_is_done_terminal_with_failures():
    # The loop must stop even when some tasks failed permanently, otherwise
    # it would spin forever printing "No runnable tasks".
    o = _make_orchestrator()
    o.tasks = [
        TaskModel(id="1", description="x", status="completed"),
        TaskModel(id="2", description="y", status="failed", retries=3),
    ]
    assert o._is_done() is True


async def test_emit_uses_sink():
    captured: list[str] = []

    async def sink(msg: str) -> None:
        captured.append(msg)

    o = _make_orchestrator()
    o._log_sink = sink
    await o._emit("hi")
    assert captured == ["hi"]


async def test_run_halts_on_success():
    original = global_settings.max_cycles
    global_settings.max_cycles = 5
    try:
        o = _make_orchestrator()
        o.planner = FakePlanner(Plan(tasks=[TaskModel(id="1", description="x")], rationale="r"))
        o.tester = FakeTester(TestResult(passed=True, summary="ok"))
        await o.run()
        assert o.cycles == 1
        assert o.tasks[0].status == "completed"
    finally:
        global_settings.max_cycles = original


async def test_run_respects_max_cycles():
    original = global_settings.max_cycles
    global_settings.max_cycles = 2
    try:
        # Tester always fails so the goal never completes, but max_cycles must halt it.
        o = _make_orchestrator()
        o.planner = FakePlanner(Plan(tasks=[TaskModel(id="1", description="x")], rationale="r"))
        o.tester = FakeTester(TestResult(passed=False, summary="nope", failures=["f"]))
        await o.run()
        assert o.cycles == 2
    finally:
        global_settings.max_cycles = original
