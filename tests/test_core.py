import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_plan_with_dependencies():
    """Test plan with task dependencies."""
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="Setup", dependencies=[]),
            TaskModel(id="2", description="Build on setup", dependencies=["1"]),
        ],
        rationale="Setup first, then build",
    )
    assert len(plan.tasks) == 2
    assert plan.tasks[1].dependencies == ["1"]


def test_task_model_default_values():
    """Test TaskModel default values."""
    task = TaskModel(id="1", description="test")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_plan_empty_tasks():
    """Test plan with no tasks."""
    plan = Plan(tasks=[], rationale="nothing to do")
    assert plan.tasks == []


def test_test_result_with_failures():
    """Test TestResult with failures."""
    result = TestResult(
        passed=False,
        summary="3 tests failed",
        failures=["test_a failed", "test_b failed"],
    )
    assert result.passed is False
    assert len(result.failures) == 2


class TestOrchestrator:
    """Tests for the Orchestrator class."""

    def test_orchestrator_init(self):
        """Test orchestrator initialization."""
        orch = Orchestrator(goal="test goal")
        assert orch.goal == "test goal"
        assert orch.cycles == 0
        assert orch.current_plan is None

    def test_is_done_no_plan(self):
        """Test _is_done returns False when no plan exists."""
        orch = Orchestrator(goal="test")
        assert orch._is_done() is False

    def test_is_done_with_completed_tasks(self):
        """Test _is_done returns True when all tasks completed."""
        orch = Orchestrator(goal="test")
        orch.current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="completed"),
            ],
            rationale="test",
        )
        assert orch._is_done() is True

    def test_is_done_with_failed_tasks(self):
        """Test _is_done returns False when tasks failed."""
        orch = Orchestrator(goal="test")
        orch.current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="failed"),
            ],
            rationale="test",
        )
        assert orch._is_done() is False

    def test_is_done_with_pending_tasks(self):
        """Test _is_done returns False when tasks are pending."""
        orch = Orchestrator(goal="test")
        orch.current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="pending"),
            ],
            rationale="test",
        )
        assert orch._is_done() is False

    def test_is_done_empty_tasks(self):
        """Test _is_done returns True when plan has no tasks."""
        orch = Orchestrator(goal="test")
        orch.current_plan = Plan(tasks=[], rationale="empty")
        assert orch._is_done() is True

    def test_get_tasks_no_plan(self):
        """Test _get_tasks returns empty list when no plan."""
        orch = Orchestrator(goal="test")
        assert orch._get_tasks() == []

    def test_get_tasks_with_plan(self):
        """Test _get_tasks returns tasks from current plan."""
        orch = Orchestrator(goal="test")
        tasks = [TaskModel(id="1", description="test")]
        orch.current_plan = Plan(tasks=tasks, rationale="test")
        assert orch._get_tasks() == tasks


class TestWorkerAgent:
    """Tests for the WorkerAgent class."""

    def test_worker_init(self):
        """Test worker initialization."""
        from furrow.agents.worker import WorkerAgent

        task = TaskModel(id="1", description="test task")
        worker = WorkerAgent(task=task)
        assert worker.task == task

    def test_parse_file_changes(self):
        """Test parsing file changes from LLM response."""
        from furrow.agents.worker import WorkerAgent

        task = TaskModel(id="1", description="test")
        worker = WorkerAgent(task=task)

        response = """
        Here's the implementation:

        FILE: src/main.py
        def hello():
            return "world"

        FILE: tests/test_main.py
        def test_hello():
            assert hello() == "world"
        """

        changes = worker._parse_file_changes(response)
        assert "src/main.py" in changes
        assert "tests/test_main.py" in changes
        assert 'def hello():' in changes["src/main.py"]
        assert 'def test_hello():' in changes["tests/test_main.py"]

    def test_parse_file_changes_empty(self):
        """Test parsing empty response."""
        from furrow.agents.worker import WorkerAgent

        task = TaskModel(id="1", description="test")
        worker = WorkerAgent(task=task)
        assert worker._parse_file_changes("no changes here") == {}

    def test_build_prompt(self):
        """Test prompt building with context."""
        from furrow.agents.worker import WorkerAgent

        task = TaskModel(id="1", description="Add feature", files=["src/app.py"])
        worker = WorkerAgent(task=task)

        prompt = worker._build_prompt("current file content")
        assert "Add feature" in prompt
        assert "src/app.py" in prompt
        assert "current file content" in prompt


class TestSettings:
    """Tests for Settings configuration."""

    def test_default_settings(self):
        """Test default settings values."""
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0

    def test_settings_from_env(self, monkeypatch):
        """Test settings loaded from environment variables."""
        monkeypatch.setenv("FURROW_PROVIDER", "openai")
        monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "10")
        s = Settings()
        assert s.provider == Provider.OPENAI
        assert s.max_parallel_tasks == 10


@pytest.mark.asyncio
async def test_orchestrator_with_mock_cycle():
    """Test orchestrator with mocked cycle."""
    output_messages = []

    async def mock_output(msg):
        output_messages.append(msg)

    orch = Orchestrator(goal="test", on_output=mock_output)

    # Mock the planner to return a simple plan
    mock_plan = Plan(
        tasks=[TaskModel(id="1", description="task 1", status="completed")],
        rationale="mock plan",
    )
    orch.planner.plan = AsyncMock(return_value=mock_plan)
    orch._execute_tasks_with_dependencies = AsyncMock(return_value=["done"])

    # Run one cycle manually
    await orch._cycle()

    assert orch.cycles == 0  # cycles incremented in run(), not _cycle()
    assert orch.current_plan == mock_plan
    assert len(output_messages) > 0
