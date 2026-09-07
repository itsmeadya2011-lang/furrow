from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


class TestOrchestratorTaskTracking:
    """Tests for the orchestrator task tracking bug fix."""

    def test_get_tasks_returns_current_tasks(self):
        """_get_tasks() should return the stored current plan tasks."""
        client = MagicMock()
        orchestrator = Orchestrator(goal="test", client=client)
        tasks = [TaskModel(id="1", description="a"), TaskModel(id="2", description="b")]
        orchestrator._current_tasks = tasks
        assert orchestrator._get_tasks() == tasks

    def test_is_done_true_when_all_completed(self):
        """_is_done() should return True when all tasks are completed."""
        client = MagicMock()
        orchestrator = Orchestrator(goal="test", client=client)
        orchestrator._current_tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ]
        assert orchestrator._is_done() is True

    def test_is_done_false_when_pending(self):
        """_is_done() should return False when any task is pending."""
        client = MagicMock()
        orchestrator = Orchestrator(goal="test", client=client)
        orchestrator._current_tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="pending"),
        ]
        assert orchestrator._is_done() is False

    def test_is_done_false_when_failed(self):
        """_is_done() should return False when any task is failed."""
        client = MagicMock()
        orchestrator = Orchestrator(goal="test", client=client)
        orchestrator._current_tasks = [
            TaskModel(id="1", description="a", status="failed"),
        ]
        assert orchestrator._is_done() is False

    def test_is_done_true_when_no_tasks(self):
        """_is_done() should return True when there are no tasks."""
        client = MagicMock()
        orchestrator = Orchestrator(goal="test", client=client)
        orchestrator._current_tasks = []
        assert orchestrator._is_done() is True


class TestWorkerAgent:
    """Tests for the worker agent file-writing behavior."""

    def test_parse_changes_extracts_files(self):
        """_parse_changes should extract file paths and content from markdown blocks."""
        client = MagicMock()
        task = TaskModel(id="1", description="add auth", files=["src/auth.py"])
        worker = WorkerAgent(task=task, client=client)
        response = """
Some intro text.

### src/auth.py
```python
def login(): pass
```

### tests/test_auth.py
```python
def test_login(): pass
```

Summary: added auth.
"""
        changes = worker._parse_changes(response)
        assert "src/auth.py" in changes
        assert "def login(): pass" in changes["src/auth.py"]
        assert "tests/test_auth.py" in changes
        assert "def test_login(): pass" in changes["tests/test_auth.py"]

    def test_parse_changes_ignores_non_matching(self):
        """_parse_changes should return empty dict if no file blocks are found."""
        client = MagicMock()
        task = TaskModel(id="1", description="do thing")
        worker = WorkerAgent(task=task, client=client)
        response = "Just some text without file blocks."
        assert worker._parse_changes(response) == {}

    @pytest.mark.asyncio
    async def test_run_writes_files(self, tmp_path: Path):
        """WorkerAgent.run() should write parsed files and return a summary."""
        client = MagicMock()
        client.complete = AsyncMock(return_value="""
### src/auth.py
```python
def login(): pass
```
""")
        client.write_file = AsyncMock()

        task = TaskModel(id="1", description="add auth", files=["src/auth.py"])
        worker = WorkerAgent(task=task, client=client)

        async def patched_write(path, content):
            p = tmp_path / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)

        client.write_file = patched_write

        result = await worker.run()
        assert "Wrote 1 file(s)" in result
        assert "src/auth.py" in result
        written = (tmp_path / "src/auth.py").read_text()
        assert "def login(): pass" in written
