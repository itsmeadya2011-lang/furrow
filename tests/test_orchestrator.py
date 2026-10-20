import asyncio

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _client(settings=None):
    class C:
        pass

    c = C()
    c.settings = settings or Settings()
    return c


def _plan(tasks):
    return Plan(tasks=tasks, rationale="r")


def test_regression_stalls(monkeypatch):
    class P:
        def __init__(self, client=None):
            pass

        async def plan(self, goal):
            return _plan([TaskModel(id="t1", description="d")])

    class W:
        def __init__(self, task=None, client=None):
            self.task = task

        async def run(self):
            raise RuntimeError("boom")

    class T:
        def __init__(self, client=None):
            pass

        async def run(self, goal, tasks):
            return TestResult(passed=False, summary="fail", failures=["x"])

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent", P)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", W)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", T)

    events = []

    async def on_event(e):
        events.append(e)

    orch = Orchestrator(goal="g", client=_client(), on_event=on_event)
    asyncio.run(asyncio.wait_for(orch.run(), timeout=30))
    assert orch.cycles > 1 and orch.cycles <= 6
    assert any(e.startswith("stalled:") for e in events)


def test_success(monkeypatch):
    class P:
        def __init__(self, client=None):
            pass

        async def plan(self, goal):
            return _plan([TaskModel(id="t1", description="d")])

    class W:
        def __init__(self, task=None, client=None):
            pass

        async def run(self):
            return "ok"

    class T:
        def __init__(self, client=None):
            pass

        async def run(self, goal, tasks):
            return TestResult(passed=True, summary="ok")

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent", P)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", W)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", T)

    events = []

    async def on_event(e):
        events.append(e)

    orch = Orchestrator(goal="g", client=_client(), on_event=on_event)
    asyncio.run(orch.run())
    assert orch.cycles == 1
    assert "goal complete" in events


def test_is_done():
    orch = Orchestrator(goal="g", client=_client())
    assert orch._is_done() is False
    orch.plan = _plan([])
    assert orch._is_done() is True
    orch.plan = _plan([TaskModel(id="a", description="d", status="failed")])
    assert orch._is_done() is False
    orch.plan = _plan([TaskModel(id="a", description="d", status="completed")])
    assert orch._is_done() is True


def test_goal_preserved(monkeypatch):
    class P:
        def __init__(self, client=None):
            pass

        async def plan(self, goal):
            return _plan([TaskModel(id="t1", description="d")])

    class W:
        def __init__(self, task=None, client=None):
            pass

        async def run(self):
            return "ok"

    class T:
        def __init__(self, client=None):
            pass

        async def run(self, goal, tasks):
            return TestResult(passed=False, summary="fail", failures=["remediation"])

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent", P)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", W)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", T)

    orch = Orchestrator(goal="ORIGINAL", client=_client(), on_event=None)
    asyncio.run(orch.run())
    assert orch.goal == "ORIGINAL"
    assert orch.focus == "remediation"


def test_parallel_cap(monkeypatch):
    settings = Settings(max_parallel_tasks=2, max_cycles=1)
    client = _client(settings)

    class P:
        def __init__(self, client=None):
            pass

        async def plan(self, goal):
            return _plan([TaskModel(id=f"t{i}", description="d") for i in range(4)])

    peak = {"v": 0}
    cur = {"v": 0}

    class W:
        def __init__(self, task=None, client=None):
            self.task = task

        async def run(self):
            cur["v"] += 1
            peak["v"] = max(peak["v"], cur["v"])
            await asyncio.sleep(0.05)
            cur["v"] -= 1
            return "ok"

    class T:
        def __init__(self, client=None):
            pass

        async def run(self, goal, tasks):
            return TestResult(passed=True, summary="ok")

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent", P)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", W)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", T)

    orch = Orchestrator(goal="g", client=client, on_event=None)
    asyncio.run(orch.run())
    assert peak["v"] <= 2
    assert peak["v"] > 1


def test_deps(monkeypatch):
    settings = Settings(max_cycles=1)
    client = _client(settings)

    class P:
        def __init__(self, client=None):
            pass

        async def plan(self, goal):
            return _plan(
                [
                    TaskModel(id="a", description="d"),
                    TaskModel(id="b", description="d", dependencies=["a"]),
                ]
            )

    order = []

    class W:
        def __init__(self, task=None, client=None):
            self.task = task

        async def run(self):
            if self.task.id == "a":
                order.append("a_start")
                await asyncio.sleep(0.05)
                order.append("a_end")
            else:
                order.append("b_start")
                await asyncio.sleep(0.01)
                order.append("b_end")
            return "ok"

    class T:
        def __init__(self, client=None):
            pass

        async def run(self, goal, tasks):
            return TestResult(passed=True, summary="ok")

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent", P)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", W)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", T)

    orch = Orchestrator(goal="g", client=client, on_event=None)
    asyncio.run(orch.run())
    assert order.index("b_start") > order.index("a_end")


def test_dep_cycle(monkeypatch):
    settings = Settings(max_cycles=1)
    client = _client(settings)

    class P:
        def __init__(self, client=None):
            pass

        async def plan(self, goal):
            return _plan(
                [
                    TaskModel(id="a", description="d", dependencies=["b"]),
                    TaskModel(id="b", description="d", dependencies=["a"]),
                ]
            )

    class W:
        def __init__(self, task=None, client=None):
            pass

        async def run(self):
            return "ok"

    class T:
        def __init__(self, client=None):
            pass

        async def run(self, goal, tasks):
            return TestResult(passed=True, summary="ok")

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent", P)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", W)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", T)

    orch = Orchestrator(goal="g", client=client, on_event=None)
    asyncio.run(asyncio.wait_for(orch.run(), timeout=10))


def test_max_cycles(monkeypatch):
    settings = Settings(max_cycles=2)
    client = _client(settings)

    class P:
        def __init__(self, client=None):
            pass

        async def plan(self, goal):
            return _plan([TaskModel(id="t1", description="d")])

    class W:
        def __init__(self, task=None, client=None):
            pass

        async def run(self):
            return "ok"

    class T:
        def __init__(self, client=None):
            pass

        async def run(self, goal, tasks):
            return TestResult(passed=False, summary="fail", failures=["x"])

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent", P)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", W)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", T)

    orch = Orchestrator(goal="g", client=client, on_event=None)
    asyncio.run(orch.run())
    assert orch.cycles <= 2


def test_events(monkeypatch):
    class P:
        def __init__(self, client=None):
            pass

        async def plan(self, goal):
            return _plan([TaskModel(id="t1", description="d")])

    class W:
        def __init__(self, task=None, client=None):
            pass

        async def run(self):
            return "ok"

    class T:
        def __init__(self, client=None):
            pass

        async def run(self, goal, tasks):
            return TestResult(passed=True, summary="ok")

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent", P)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", W)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", T)

    events = []

    async def on_event(e):
        events.append(e)

    orch = Orchestrator(goal="g", client=_client(), on_event=on_event)
    asyncio.run(orch.run())
    assert "cycle 1 start" in events
    assert "plan ready: 1 task(s)" in events


# structlog log.exception misuse in orchestrator._emit
def test_callback_raises(monkeypatch):
    class P:
        def __init__(self, client=None):
            pass

        async def plan(self, goal):
            return _plan([TaskModel(id="t1", description="d")])

    class W:
        def __init__(self, task=None, client=None):
            pass

        async def run(self):
            return "ok"

    class T:
        def __init__(self, client=None):
            pass

        async def run(self, goal, tasks):
            return TestResult(passed=True, summary="ok")

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent", P)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", W)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", T)

    async def on_event(e):
        raise RuntimeError("cb boom")

    orch = Orchestrator(goal="g", client=_client(), on_event=on_event)
    asyncio.run(orch.run())
    assert orch.cycles == 1
