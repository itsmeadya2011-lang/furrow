import asyncio
import json

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult


class FakeLLMClient:
    """Duck-typed LLMClient returning canned responses in order."""

    def __init__(self, responses, settings=None):
        self.responses = list(responses)
        self.settings = settings or Settings()
        self.prompts: list[str] = []

    async def complete(self, prompt, system="", model=None):
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else ""


class FakePlanner:
    def __init__(self, client=None):
        self.client = client
        self.goals: list[str] = []
        self._responses: list[Plan] = []

    async def plan(self, goal):
        self.goals.append(goal)
        return self._responses.pop(0)

    def queue(self, plan):
        self._responses.append(plan)


class FakeWorker:
    def __init__(self, task, client=None):
        self.task = task

    async def run(self):
        return "ok"


class FakeTester:
    def __init__(self, client=None):
        self.client = client
        self._responses = []

    def queue(self, result):
        self._responses.append(result)

    async def run(self, goal, tasks):
        return self._responses.pop(0)


@pytest.fixture
def orch_module(monkeypatch):
    import furrow.core.orchestrator as orch_mod

    return orch_mod


@pytest.fixture
def patched_agents(monkeypatch, orch_module):
    planner = FakePlanner(client=None)
    tester = FakeTester(client=None)

    monkeypatch.setattr(orch_module, "PlannerAgent", lambda client=None: planner)
    monkeypatch.setattr(orch_module, "TesterAgent", lambda client=None: tester)
    monkeypatch.setattr(orch_module, "WorkerAgent", FakeWorker)

    return planner, tester


async def test_run_completes_when_tests_pass(monkeypatch, patched_agents):
    planner, tester = patched_agents
    planner.queue(Plan(tasks=[TaskModel(id="1", description="a"), TaskModel(id="2", description="b")], rationale="ok"))
    tester.queue(TestResult(passed=True, summary="ok", failures=[]))

    s = Settings(max_cycles=0, max_parallel_tasks=5)
    fake = FakeLLMClient([], settings=s)

    from furrow.core.orchestrator import Orchestrator

    captured = []

    def on_event(payload):
        captured.append(payload)

    orch = Orchestrator(goal="g", client=fake, settings=s, on_event=on_event)
    await orch.run()

    events = [c["event"] for c in captured]
    for required in ["start", "cycle_start", "plan", "task_start", "task_complete", "test_result", "complete"]:
        assert required in events, f"missing event {required}"

    for payload in captured:
        json.dumps(payload)

    summary = orch.summary()
    assert summary["cycles"] == 1
    assert summary["completed"] == 2


async def test_failed_tests_trigger_fix_cycle(monkeypatch, patched_agents):
    planner, tester = patched_agents
    planner.queue(Plan(tasks=[TaskModel(id="1", description="a")], rationale="ok"))
    planner.queue(Plan(tasks=[], rationale="done"))
    tester.queue(TestResult(passed=False, summary="fail", failures=["assert 1 == 2"]))
    tester.queue(TestResult(passed=True, summary="ok", failures=[]))

    s = Settings(max_cycles=0, max_parallel_tasks=5)
    fake = FakeLLMClient([], settings=s)

    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="original goal", client=fake, settings=s)
    await orch.run()

    assert orch.cycles == 2
    assert "assert 1 == 2" in planner.goals[1]
    assert "original goal" in planner.goals[1]


async def test_max_cycles_halts(monkeypatch, patched_agents):
    planner, tester = patched_agents
    planner.queue(Plan(tasks=[TaskModel(id="1", description="a")], rationale="ok"))
    planner.queue(Plan(tasks=[TaskModel(id="1", description="a")], rationale="ok"))
    tester.queue(TestResult(passed=False, summary="fail", failures=["x"]))
    tester.queue(TestResult(passed=False, summary="fail", failures=["x"]))

    s = Settings(max_cycles=2, max_parallel_tasks=5)
    fake = FakeLLMClient([], settings=s)

    from furrow.core.orchestrator import Orchestrator

    captured = []

    def on_event(payload):
        captured.append(payload)

    orch = Orchestrator(goal="g", client=fake, settings=s, on_event=on_event)
    await orch.run()

    cycle_starts = [c for c in captured if c["event"] == "cycle_start"]
    assert len(cycle_starts) == 2
    assert any(c["event"] == "max_cycles_reached" for c in captured)


async def test_parallelism_limited(monkeypatch, patched_agents):
    planner, tester = patched_agents
    for _ in range(3):
        planner.queue(Plan(tasks=[TaskModel(id="1", description="a")], rationale="ok"))
    tester.queue(TestResult(passed=True, summary="ok", failures=[]))

    s = Settings(max_cycles=0, max_parallel_tasks=1)
    fake = FakeLLMClient([], settings=s)

    from furrow.core.orchestrator import Orchestrator

    counter = {"current": 0, "max": 0}

    class CountingWorker:
        def __init__(self, task, client=None):
            self.task = task

        async def run(self):
            counter["current"] += 1
            counter["max"] = max(counter["max"], counter["current"])
            await asyncio.sleep(0.05)
            counter["current"] -= 1
            return "ok"

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", CountingWorker)

    orch = Orchestrator(goal="g", client=fake, settings=s)
    await orch.run()

    assert counter["max"] == 1


async def test_worker_exception_marks_task_failed(monkeypatch, patched_agents):
    planner, tester = patched_agents
    planner.queue(Plan(tasks=[TaskModel(id="1", description="a")], rationale="ok"))
    planner.queue(Plan(tasks=[], rationale="done"))
    tester.queue(TestResult(passed=True, summary="ok", failures=[]))

    s = Settings(max_cycles=0, max_parallel_tasks=5)
    fake = FakeLLMClient([], settings=s)

    from furrow.core.orchestrator import Orchestrator

    captured = []

    def on_event(payload):
        captured.append(payload)

    class FailingWorker:
        def __init__(self, task, client=None):
            self.task = task

        async def run(self):
            raise RuntimeError("boom")

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FailingWorker)

    orch = Orchestrator(goal="g", client=fake, settings=s, on_event=on_event)
    await orch.run()

    failed_events = [c for c in captured if c["event"] == "task_failed"]
    assert len(failed_events) == 1
    assert failed_events[0]["error"] == "boom"
    assert failed_events[0]["task"]["status"] == "failed"


async def test_no_tasks_planned_completes(monkeypatch, patched_agents):
    planner, _tester = patched_agents
    planner.queue(Plan(tasks=[], rationale="nothing to do"))

    s = Settings(max_cycles=0, max_parallel_tasks=5)
    fake = FakeLLMClient([], settings=s)

    from furrow.core.orchestrator import Orchestrator

    captured = []

    def on_event(payload):
        captured.append(payload)

    orch = Orchestrator(goal="g", client=fake, settings=s, on_event=on_event)
    await orch.run()

    events = [c["event"] for c in captured]
    assert "no_tasks" in events
    assert "complete" in events
    assert orch.cycles == 1