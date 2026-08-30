"""Tests for Furrow core components."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class TestOrchestrator:
    """Tests for the Orchestrator class."""

    def test_is_done_no_tasks(self):
        """When no plan exists, _is_done returns True."""
        client = MagicMock()
        client.settings.max_cycles = 0
        orch = Orchestrator(goal="test", client=client)
        assert orch._is_done() is True

    def test_is_done_all_completed(self):
        """When all tasks are completed, _is_done returns True."""
        client = MagicMock()
        client.settings.max_cycles = 0
        orch = Orchestrator(goal="test", client=client)
        orch.current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="completed"),
            ],
            rationale="test",
        )
        assert orch._is_done() is True

    def test_is_done_with_failures(self):
        """When any task failed, _is_done returns False."""
        client = MagicMock()
        client.settings.max_cycles = 0
        orch = Orchestrator(goal="test", client=client)
        orch.current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="failed"),
            ],
            rationale="test",
        )
        assert orch._is_done() is False

    def test_is_done_partial_completion(self):
        """When some tasks are pending, _is_done returns False."""
        client = MagicMock()
        client.settings.max_cycles = 0
        orch = Orchestrator(goal="test", client=client)
        orch.current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="pending"),
            ],
            rationale="test",
        )
        assert orch._is_done() is False

    def test_get_tasks_empty(self):
        """_get_tasks returns empty list when no plan."""
        client = MagicMock()
        orch = Orchestrator(goal="test", client=client)
        assert orch._get_tasks() == []

    def test_get_tasks_returns_plan_tasks(self):
        """_get_tasks returns the current plan's tasks."""
        client = MagicMock()
        orch = Orchestrator(goal="test", client=client)
        tasks = [TaskModel(id="1", description="a")]
        orch.current_plan = Plan(tasks=tasks, rationale="test")
        assert orch._get_tasks() == tasks

    @pytest.mark.asyncio
    async def test_max_cycles_enforcement(self):
        """Orchestrator stops after max_cycles."""
        client = MagicMock()
        client.settings.max_cycles = 2
        client.settings.planner_model = "test"
        client.settings.worker_model = "test"
        client.settings.tester_model = "test"
        client.settings.workspace = MagicMock()

        orch = Orchestrator(goal="test", client=client)

        # Mock the planner to return a plan with one task
        mock_plan = Plan(
            tasks=[TaskModel(id="1", description="do something")],
            rationale="test",
        )
        orch.planner.plan = AsyncMock(return_value=mock_plan)

        # Mock the worker
        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
             patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            MockWorker.return_value.run = AsyncMock(return_value="done")
            MockTester.return_value.run = AsyncMock(
                return_value=TestResult(passed=True, summary="ok")
            )

            await orch.run()

        # Should run max_cycles times (2), then increment to 3 and break
        assert orch.cycles == 3

    @pytest.mark.asyncio
    async def test_no_tasks_halts(self):
        """Orchestrator halts when planner returns no tasks."""
        client = MagicMock()
        client.settings.max_cycles = 0
        client.settings.planner_model = "test"
        client.settings.workspace = MagicMock()

        orch = Orchestrator(goal="test", client=client)

        # Mock the planner to return empty plan
        mock_plan = Plan(tasks=[], rationale="nothing to do")
        orch.planner.plan = AsyncMock(return_value=mock_plan)

        await orch.run()

        # Should complete after one cycle with no tasks
        assert orch.cycles == 1


class TestPlan:
    """Tests for the Plan model."""

    def test_plan_creation(self):
        """Plan can be created with tasks."""
        p = Plan(
            tasks=[TaskModel(id="1", description="do thing")],
            rationale="ok",
        )
        assert len(p.tasks) == 1
        assert p.tasks[0].description == "do thing"
        assert p.rationale == "ok"

    def test_plan_empty_tasks(self):
        """Plan can have empty tasks list."""
        p = Plan(tasks=[], rationale="empty")
        assert p.tasks == []

    def test_task_default_status(self):
        """TaskModel defaults to pending status."""
        t = TaskModel(id="1", description="test")
        assert t.status == "pending"
        assert t.result is None


class TestTestResult:
    """Tests for the TestResult model."""

    def test_test_result_passed(self):
        """TestResult can represent passing tests."""
        t = TestResult(passed=True, summary="All tests passed", failures=[])
        assert t.passed is True
        assert t.summary == "All tests passed"

    def test_test_result_failed(self):
        """TestResult can represent failing tests."""
        t = TestResult(passed=False, summary="Some tests failed", failures=["test_a"])
        assert t.passed is False
        assert len(t.failures) == 1

    def test_test_result_default_failures(self):
        """TestResult failures defaults to empty list."""
        t = TestResult(passed=True, summary="ok")
        assert t.failures == []


class TestWorkerFileParsing:
    """Tests for the WorkerAgent file parsing logic."""

    def test_parse_files_extracts_content(self):
        """Worker correctly parses file contents from LLM response."""
        from furrow.agents.worker import WorkerAgent

        client = MagicMock()
        task = TaskModel(id="1", description="test", files=["test.py"])
        worker = WorkerAgent(task=task, client=client)

        response = """## Files to modify

### test.py
```python
def hello():
    return "world"
```

## Summary
Created hello function
"""
        result = worker._parse_files(response)
        assert "test.py" in result
        assert 'def hello():' in result["test.py"]

    def test_parse_files_multiple_files(self):
        """Worker correctly parses multiple files."""
        from furrow.agents.worker import WorkerAgent

        client = MagicMock()
        task = TaskModel(id="1", description="test")
        worker = WorkerAgent(task=task, client=client)

        response = """## Files to modify

### a.py
```python
# file a
```

### b.py
```python
# file b
```

## Summary
Modified two files
"""
        result = worker._parse_files(response)
        assert len(result) == 2
        assert "a.py" in result
        assert "b.py" in result

    def test_parse_files_empty(self):
        """Worker returns empty dict when no files section."""
        from furrow.agents.worker import WorkerAgent

        client = MagicMock()
        task = TaskModel(id="1", description="test")
        worker = WorkerAgent(task=task, client=client)

        result = worker._parse_files("No files here")
        assert result == {}

    def test_extract_summary(self):
        """Worker extracts summary from response."""
        from furrow.agents.worker import WorkerAgent

        client = MagicMock()
        task = TaskModel(id="1", description="test")
        worker = WorkerAgent(task=task, client=client)

        response = """## Files to modify

### test.py
```python
# test
```

## Summary
Did some work
"""
        summary = worker._extract_summary(response, {"test.py": "content"})
        assert summary == "Did some work"

    def test_extract_summary_fallback(self):
        """Worker provides fallback summary when none in response."""
        from furrow.agents.worker import WorkerAgent

        client = MagicMock()
        task = TaskModel(id="1", description="test")
        worker = WorkerAgent(task=task, client=client)

        summary = worker._extract_summary("no summary", {"a.py": "", "b.py": ""})
        assert "Modified 2 file(s)" in summary


class TestPlannerFileTree:
    """Tests for the PlannerAgent file tree filtering."""

    def test_excluded_patterns(self):
        """Excluded patterns are filtered from file tree."""
        from furrow.agents.planner import _is_excluded

        assert _is_excluded("src/__pycache__/module.py") is True
        assert _is_excluded(".git/HEAD") is True
        assert _is_excluded("node_modules/lodash/index.js") is True
        assert _is_excluded(".venv/bin/python") is True
        assert _is_excluded("venv/lib/python3.10/site-packages/foo.py") is True
        assert _is_excluded("dist/build.js") is True
        assert _is_excluded("build/output.o") is True
        assert _is_excluded("mypackage.egg-info/PKG-INFO") is True

    def test_non_excluded_patterns(self):
        """Normal source files are not excluded."""
        from furrow.agents.planner import _is_excluded

        assert _is_excluded("src/main.py") is False
        assert _is_excluded("tests/test_foo.py") is False
        assert _is_excluded("README.md") is False
        assert _is_excluded("pyproject.toml") is False
        assert _is_excluded("src/utils/helpers.py") is False
