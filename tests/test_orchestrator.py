import io

import furrow.core.orchestrator as orch_mod
from furrow.core.orchestrator import Orchestrator
from furrow.config import Plan, TaskModel, TestResult, Settings
from furrow.llm import LLMClient, Provider
from rich.console import Console


class _FakeWorker:
    def __init__(self, task, client=None, **kwargs):
        self.task = task

    async def run(self):
        self.task.status = "completed"
        self.task.result = "fake done"
        return "fake done"


class _PassTester:
    def __init__(self, *a, **k):
        pass

    async def run(self, goal, tasks):
        return TestResult(passed=True, summary="ok", failures=[])


class _FailTester:
    def __init__(self, *a, **k):
        pass

    async def run(self, goal, tasks):
        return TestResult(passed=False, summary="fail", failures=["thing broke"])


class _FakePlanner:
    def __init__(self, *a, **k):
        self.calls = 0

    async def plan(self, goal):
        self.calls += 1
        return Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")


def _settings(tmp_path, **overrides):
    kwargs = {
        "provider": Provider.ANTHROPIC,
        "anthropic_api_key": "x",
        "workspace": tmp_path,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


def _patch(monkeypatch, tester_cls=_PassTester):
    monkeypatch.setattr(orch_mod, "PlannerAgent", _FakePlanner)
    monkeypatch.setattr(orch_mod, "TesterAgent", tester_cls)
    monkeypatch.setattr(orch_mod, "WorkerAgent", lambda **kw: _FakeWorker(**kw))


def _make(settings, monkeypatch, tester_cls=_PassTester):
    _patch(monkeypatch, tester_cls=tester_cls)
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, highlight=False, width=100)
    orch = Orchestrator(goal="g", client=LLMClient(settings=settings), console=console)
    return orch, buf


async def test_run_stops_when_tests_pass(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    orch, buf = _make(settings, monkeypatch)
    await orch.run()
    assert orch.cycles == 1
    assert orch.tasks[0].status == "completed"
    assert orch.tasks[0].result == "fake done"
    assert "Tests passed" in buf.getvalue()


async def test_run_default_console_does_not_crash(tmp_path, monkeypatch):
    _patch(monkeypatch)
    settings = _settings(tmp_path)
    orch = Orchestrator(goal="g", client=LLMClient(settings=settings))
    assert orch.console is not None
    await orch.run()
    assert orch.cycles == 1


async def test_run_respects_max_cycles(tmp_path, monkeypatch):
    settings = _settings(tmp_path, max_cycles=2)
    orch, buf = _make(settings, monkeypatch, tester_cls=_FailTester)
    await orch.run()
    assert orch.cycles == 2
    assert orch.tasks[0].status == "completed"
    assert "Reached max_cycles" in buf.getvalue()


async def test_tasks_persisted_across_cycles(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    orch, buf = _make(settings, monkeypatch)
    await orch.run()
    assert len(orch.tasks) == 1
    assert orch.tasks[0].id == "1"
