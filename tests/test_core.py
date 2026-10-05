import pytest
from furrow.config import Plan, TaskModel
from furrow.config import TestResult as FurrowTestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = FurrowTestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_done_with_empty_tasks():
    """An orchestrator with no tasks and no cycle cap should not halt immediately."""
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="do nothing")
    assert orch._is_done() is False


def test_orchestrator_done_when_all_completed():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="x")
    orch.tasks = [TaskModel(id="1", description="a", status="completed")]
    assert orch._is_done() is True


def test_orchestrator_not_done_when_failed():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="x")
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orch._is_done() is False


def test_orchestrator_max_cycles_halts():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="x")
    orch.max_cycles = 3
    orch.cycles = 3
    assert orch._is_done() is True


def test_worker_writes_files(tmp_path, monkeypatch):
    """WorkerAgent should parse code blocks and write them to disk."""
    import asyncio

    from furrow.agents.worker import WorkerAgent
    from furrow.config import TaskModel

    monkeypatch.chdir(tmp_path)

    written: dict[str, str] = {}

    class FakeClient:
        class settings:
            worker_model = "fake"

        async def complete(self, prompt, model=None):
            return (
                "Here is the implementation:\n"
                "```python src/auth.py\nprint('hi')\n```\n"
                "```python\nx = 1\n```\n"
            )

        async def write_file(self, path, content):
            written[path] = content

    task = TaskModel(id="1", description="add auth", files=["fallback.py"])
    agent = WorkerAgent(task=task, client=FakeClient())
    result = asyncio.run(agent.run())

    assert "src/auth.py" in written
    assert written["src/auth.py"].strip() == "print('hi')"
    # Second block has no path tag and the files list is exhausted, so a
    # generated name is used.
    assert len(written) == 2
    assert list(written.values())[1].strip() == "x = 1"
    assert "Wrote 2 file(s)" in result


def test_worker_no_code_blocks(tmp_path, monkeypatch):
    import asyncio

    from furrow.agents.worker import WorkerAgent
    from furrow.config import TaskModel

    monkeypatch.chdir(tmp_path)

    class FakeClient:
        class settings:
            worker_model = "fake"

        async def complete(self, prompt, model=None):
            return "I cannot do this task."

        async def write_file(self, path, content):
            pytest.fail("should not write any files")

    task = TaskModel(id="1", description="impossible", files=[])
    agent = WorkerAgent(task=task, client=FakeClient())
    result = asyncio.run(agent.run())
    assert "No code blocks" in result