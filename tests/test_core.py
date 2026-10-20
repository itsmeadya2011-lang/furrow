import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.agents.worker import WorkerAgent


class FakeClient:
    """Stand-in for LLMClient that records writes and returns canned completions."""

    def __init__(self) -> None:
        self.writes: dict[str, str] = {}
        self.reads: dict[str, str] = {}

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        return '{"src/foo.py": "x = 1"}'

    async def read_file(self, path: str) -> str:
        if path in self.reads:
            return self.reads[path]
        raise FileNotFoundError(path)

    async def write_file(self, path: str, content: str) -> None:
        self.writes[path] = content


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_is_done_no_tasks():
    orchestrator = Orchestrator(goal="x")
    assert orchestrator._is_done() is False


def test_is_done_all_completed():
    orchestrator = Orchestrator(goal="x")
    orchestrator.all_tasks = [TaskModel(id="1", description="a", status="completed")]
    assert orchestrator._is_done() is True


def test_is_done_one_failed():
    orchestrator = Orchestrator(goal="x")
    orchestrator.all_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orchestrator._is_done() is False


def test_is_done_pending_remaining():
    orchestrator = Orchestrator(goal="x")
    orchestrator.all_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orchestrator._is_done() is False


def test_max_cycles_unlimited_by_default():
    orchestrator = Orchestrator(goal="x")
    assert orchestrator._max_cycles_reached() is False


def test_max_cycles_reached():
    orchestrator = Orchestrator(goal="x")
    orchestrator.cycles = 5
    orchestrator.client.settings.max_cycles = 3
    assert orchestrator._max_cycles_reached() is True


def test_max_cycles_not_reached_yet():
    orchestrator = Orchestrator(goal="x")
    orchestrator.cycles = 2
    orchestrator.client.settings.max_cycles = 5
    assert orchestrator._max_cycles_reached() is False


@pytest.mark.asyncio
async def test_worker_writes_files():
    client = FakeClient()
    worker = WorkerAgent(
        task=TaskModel(id="1", description="create foo", files=["src/foo.py"]),
        client=client,
    )
    result = await worker.run()
    assert "src/foo.py" in client.writes
    assert client.writes["src/foo.py"] == "x = 1"
    assert "Wrote 1 file(s): src/foo.py" in result


@pytest.mark.asyncio
async def test_worker_reads_existing_files_for_context():
    client = FakeClient()
    client.reads = {"src/foo.py": "existing content"}
    worker = WorkerAgent(
        task=TaskModel(id="1", description="edit foo", files=["src/foo.py"]),
        client=client,
    )
    await worker.run()
    assert "src/foo.py" in client.writes
