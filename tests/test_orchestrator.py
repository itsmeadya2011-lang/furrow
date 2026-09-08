import asyncio

import pytest
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.core.orchestrator import Orchestrator


class FakeClient:
    def __init__(self, response="", files=None):
        self.response = response
        self.writes = []
        self._files = files or {}

    async def complete(self, prompt, system="", model=None):
        return self.response

    async def read_file(self, path):
        if str(path) in self._files:
            return self._files[str(path)]
        raise FileNotFoundError(str(path))

    async def write_file(self, path, content):
        self.writes.append((str(path), content))

    @property
    def settings(self):
        from furrow.config import Settings
        return Settings()


class FakePlanner:
    def __init__(self, plan):
        self._plan = plan

    async def plan(self, goal):
        return self._plan


class FakeWorker:
    def __init__(self, task, client=None):
        self.task = task
        self.client = client

    async def run(self):
        return "ok"


class FakeTester:
    def __init__(self, client=None):
        self.client = client

    async def run(self, goal, tasks):
        return TestResult(passed=True, summary="ok")


@pytest.fixture(autouse=True)
def _patch_agents(monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", FakeTester)


def test_orchestrator_task_tracking():
    plan = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="ok",
    )
    output = []
    orchestrator = Orchestrator("test goal", client=FakeClient(), on_output=output.append)
    orchestrator.planner = FakePlanner(plan)

    asyncio.run(orchestrator.run())

    tasks = orchestrator._get_tasks()
    assert len(tasks) == 1
    assert tasks[0].status == "completed"


def test_orchestrator_max_cycles(monkeypatch):
    original = settings.max_cycles
    try:
        settings.max_cycles = 2
        plan = Plan(
            tasks=[TaskModel(id="1", description="do thing")],
            rationale="keep going",
        )
        failing_result = TestResult(passed=False, summary="fail", failures=["test fail"])

        class FailingFakeWorker:
            def __init__(self, task, client=None):
                self.task = task
                self.client = client

            async def run(self):
                raise RuntimeError("worker failed")

        class FailingTester:
            def __init__(self, client=None):
                self.client = client

            async def run(self, goal, tasks):
                return failing_result

        monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FailingFakeWorker)
        monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", FailingTester)

        orchestrator = Orchestrator("test goal", client=FakeClient())
        orchestrator.planner = FakePlanner(plan)

        asyncio.run(orchestrator.run())

        assert orchestrator.cycles == 2
    finally:
        settings.max_cycles = original


def test_orchestrator_done_when_no_tasks():
    plan = Plan(tasks=[], rationale="done")
    orchestrator = Orchestrator("test goal", client=FakeClient())
    orchestrator.planner = FakePlanner(plan)

    asyncio.run(orchestrator.run())

    assert orchestrator.cycles == 1


def test_orchestrator_emit_callback():
    plan = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="ok",
    )
    output = []
    orchestrator = Orchestrator("test goal", client=FakeClient(), on_output=output.append)
    orchestrator.planner = FakePlanner(plan)

    asyncio.run(orchestrator.run())

    assert len(output) > 0
    combined = " ".join(output)
    assert "Furrow" in combined or "Cycle" in combined


async def test_worker_parses_file_blocks():
    response = "---FILE: test.txt---hello---END FILE---\nSUMMARY: wrote test"
    client = FakeClient(response=response)
    from furrow.agents.worker import WorkerAgent

    task = TaskModel(id="1", description="write file", files=[])
    w = WorkerAgent(task=task, client=client)
    result = await w.run()

    assert "Updated 1 files" in result
    assert client.writes == [("test.txt", "hello")]


async def test_worker_skips_missing_files():
    response = "---FILE: new.txt---world---END FILE---\nSUMMARY: wrote new"
    client = FakeClient(response=response, files={})
    from furrow.agents.worker import WorkerAgent

    task = TaskModel(id="1", description="write file", files=["missing.txt"])
    w = WorkerAgent(task=task, client=client)
    result = await w.run()

    assert "Updated 1 files" in result
    assert client.writes == [("new.txt", "world")]
