import pytest
from unittest.mock import AsyncMock
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# ---------------------------------------------------------------------------
# Orchestrator._is_done tests
# ---------------------------------------------------------------------------

def test_orchestrator_is_done_no_tasks():
    from furrow.core.orchestrator import Orchestrator

    orchestrator = Orchestrator(goal="test")
    orchestrator._plan = Plan(tasks=[], rationale="empty")
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_with_completed_tasks():
    from furrow.core.orchestrator import Orchestrator

    orchestrator = Orchestrator(goal="test")
    orchestrator._plan = Plan(
        tasks=[TaskModel(id="1", description="done", status="completed")],
        rationale="ok",
    )
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_with_failed_tasks():
    from furrow.core.orchestrator import Orchestrator

    orchestrator = Orchestrator(goal="test")
    orchestrator._plan = Plan(
        tasks=[TaskModel(id="1", description="fail", status="failed")],
        rationale="ok",
    )
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_with_pending_tasks():
    from furrow.core.orchestrator import Orchestrator

    orchestrator = Orchestrator(goal="test")
    orchestrator._plan = Plan(
        tasks=[TaskModel(id="1", description="pending", status="pending")],
        rationale="ok",
    )
    assert orchestrator._is_done() is False


# ---------------------------------------------------------------------------
# Orchestrator max_cycles test
# ---------------------------------------------------------------------------

def test_orchestrator_max_cycles(monkeypatch):
    from furrow.core.orchestrator import Orchestrator
    from furrow.config import settings

    monkeypatch.setattr(settings, "max_cycles", 2)

    orchestrator = Orchestrator(goal="test")
    orchestrator.planner = AsyncMock()
    orchestrator.planner.plan = AsyncMock(
        return_value=Plan(
            tasks=[TaskModel(id="1", description="pending task", status="pending")],
            rationale="ok",
        )
    )

    # Mock worker to raise exception so task stays "failed" and _is_done returns False
    class FakeWorker:
        def __init__(self, *args, **kwargs):
            pass
        async def run(self):
            raise RuntimeError("mock failure")

    # Mock tester to return passing result
    class FakeTester:
        def __init__(self, *args, **kwargs):
            pass
        async def run(self, goal, tasks):
            return TestResult(passed=True, summary="ok", failures=[])

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)

    orchestrator.tester = FakeTester()

    import asyncio
    asyncio.run(orchestrator.run())

    assert orchestrator.cycles == 2


# ---------------------------------------------------------------------------
# WorkerAgent.run writes files test
# ---------------------------------------------------------------------------

def test_worker_run_writes_files():
    from furrow.agents.worker import WorkerAgent
    from furrow.config import TaskModel

    class FakeClient:
        def __init__(self):
            from furrow.config import Settings
            self.settings = Settings()

        async def complete(self, prompt, system="", model=None):
            return '{"files": {"test.py": "print(\'hello\')"}}'

        async def read_file(self, path):
            return ""

        async def write_file(self, path, content):
            self.last_path = path
            self.last_content = content

        def list_files(self, dir):
            return []

    client = FakeClient()
    task = TaskModel(id="1", description="write a file")
    worker = WorkerAgent(task=task, client=client)

    import asyncio
    result = asyncio.run(worker.run())

    assert client.last_path == "test.py"
    assert client.last_content == "print('hello')"
    assert "Wrote 1 file(s)" in result


# ---------------------------------------------------------------------------
# Callbacks emitted test
# ---------------------------------------------------------------------------

def test_callbacks_emitted():
    from furrow.core.orchestrator import Orchestrator
    from furrow.config import Plan, TaskModel, TestResult

    class FakeClient:
        def __init__(self):
            from furrow.config import Settings
            self.settings = Settings()

        async def complete(self, prompt, system="", model=None):
            return '{"files": {"test.py": "print(1)"}}'

        async def read_file(self, path):
            return ""

        async def write_file(self, path, content):
            pass

        def list_files(self, dir):
            return []

    orchestrator = Orchestrator(goal="test", client=FakeClient())

    # Mock planner to return a plan with one task
    orchestrator.planner = AsyncMock()
    orchestrator.planner.plan = AsyncMock(
        return_value=Plan(
            tasks=[TaskModel(id="1", description="do something")],
            rationale="ok",
        )
    )

    # Mock tester to return a passing TestResult
    class FakeTester:
        async def run(self, goal, tasks):
            return TestResult(passed=True, summary="all good", failures=[])

    orchestrator.tester = FakeTester()

    callbacks = []
    orchestrator._callbacks = [callbacks.append]

    import asyncio
    asyncio.run(orchestrator._cycle())

    msg_types = [msg["type"] for msg in callbacks]
    assert "plan" in msg_types
    assert "task_done" in msg_types
    assert "test_result" in msg_types