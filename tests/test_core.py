import json
import re
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.worker import TOOL_PATTERN, WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class TestConfigModels:
    def test_task_model_creation(self):
        task = TaskModel(id="1", description="Test task", files=["src/test.py"])
        assert task.id == "1"
        assert task.description == "Test task"
        assert task.files == ["src/test.py"]
        assert task.status == "pending"

    def test_task_model_with_dependencies(self):
        task = TaskModel(id="2", description="Dependent task", dependencies=["1"])
        assert task.dependencies == ["1"]

    def test_plan_creation(self):
        tasks = [TaskModel(id="1", description="Task 1")]
        plan = Plan(tasks=tasks, rationale="Test rationale")
        assert len(plan.tasks) == 1
        assert plan.rationale == "Test rationale"

    def test_test_result_passed(self):
        result = TestResult(passed=True, summary="All good", failures=[])
        assert result.passed is True
        assert result.failures == []

    def test_test_result_failed(self):
        result = TestResult(passed=False, summary="Failed", failures=["Error 1"])
        assert result.passed is False
        assert result.failures == ["Error 1"]

    def test_provider_enum(self):
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.OPENAI == "openai"
        assert Provider.OLLAMA == "ollama"

    def test_settings_defaults(self):
        settings = Settings()
        assert settings.provider == Provider.ANTHROPIC
        assert settings.max_parallel_tasks == 5
        assert settings.max_cycles == 0


class TestOrchestrator:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "done"}')
        client.settings = MagicMock()
        client.settings.worker_model = "test-model"
        client.settings.planner_model = "test-model"
        client.settings.tester_model = "test-model"
        client.settings.max_parallel_tasks = 5
        return client

    @pytest.mark.asyncio
    async def test_task_tracking_across_cycles(self, mock_client):
        orchestrator = Orchestrator(goal="Test goal", client=mock_client)

        # Simulate two cycles
        task1 = TaskModel(id="1", description="Task 1")
        task2 = TaskModel(id="2", description="Task 2")

        orchestrator.all_tasks.append(task1)
        orchestrator.all_tasks.append(task2)

        tasks = orchestrator._get_tasks()
        assert len(tasks) == 2
        assert tasks[0].description == "Task 1"
        assert tasks[1].description == "Task 2"

    @pytest.mark.asyncio
    async def test_is_done_all_completed(self, mock_client):
        orchestrator = Orchestrator(goal="Test goal", client=mock_client)
        orchestrator.all_tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="completed"),
        ]
        assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_is_done_has_failed(self, mock_client):
        orchestrator = Orchestrator(goal="Test goal", client=mock_client)
        orchestrator.all_tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="failed"),
        ]
        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_is_done_max_cycles(self, mock_client):
        mock_client.settings.max_cycles = 3
        orchestrator = Orchestrator(goal="Test goal", client=mock_client)
        orchestrator.cycles = 3
        orchestrator.all_tasks = [
            TaskModel(id="1", description="Task 1", status="pending"),
        ]
        assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_state_persistence_roundtrip(self, mock_client):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "test_state.json"
            orchestrator = Orchestrator(
                goal="Test goal", client=mock_client, state_file=state_file
            )

            # Add some state
            orchestrator.cycles = 2
            orchestrator.all_tasks = [
                TaskModel(id="1", description="Task 1", status="completed"),
                TaskModel(id="2", description="Task 2", status="failed"),
            ]
            orchestrator.last_test_failures = ["Test failed"]
            orchestrator.completed_per_cycle = [["1"]]
            orchestrator.task_status_changes = {"1": [("pending", "completed")]}

            # Save state
            orchestrator.save_state()

            # Load state into new orchestrator
            orchestrator2 = Orchestrator(
                goal="Test goal", client=mock_client, state_file=state_file, resume=True
            )

            assert orchestrator2.cycles == 2
            assert len(orchestrator2.all_tasks) == 2
            assert orchestrator2.all_tasks[0].description == "Task 1"
            assert orchestrator2.all_tasks[0].status == "completed"
            assert orchestrator2.all_tasks[1].status == "failed"
            assert orchestrator2.last_test_failures == ["Test failed"]
            assert orchestrator2.completed_per_cycle == [["1"]]


class TestWorkerAgent:
    def test_tool_pattern_parsing(self):
        response = 'TOOL: read_file {"path": "src/test.py"}'
        matches = TOOL_PATTERN.findall(response)
        assert len(matches) == 1
        assert matches[0] == ("read_file", '{"path": "src/test.py"}')

    def test_tool_pattern_multiple(self):
        response = """TOOL: read_file {"path": "src/a.py"}
TOOL: write_file {"path": "src/b.py", "content": "hello"}"""
        matches = TOOL_PATTERN.findall(response, re.MULTILINE)
        assert len(matches) == 2

    def test_build_summary_no_changes(self):
        worker = WorkerAgent(task=TaskModel(id="1", description="Test"))
        worker._changed_files = []
        worker._run_commands = []
        assert worker._build_summary() == "No changes were made."

    def test_build_summary_with_changes(self):
        worker = WorkerAgent(task=TaskModel(id="1", description="Test"))
        worker._changed_files = ["src/a.py", "src/b.py"]
        worker._run_commands = ["pytest -q"]
        summary = worker._build_summary()
        assert "Files modified:" in summary
        assert "src/a.py" in summary
        assert "src/b.py" in summary
        assert "Commands run:" in summary
        assert "pytest -q" in summary


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_with_context(self):
        client = MagicMock()
        client.complete = AsyncMock(
            return_value='{"tasks": [{"id": "1", "description": "Fix test", "files": [], "dependencies": []}], "rationale": "Fix failing test"}'
        )
        client.settings = MagicMock()
        client.settings.planner_model = "test-model"

        from furrow.agents.planner import PlannerAgent

        planner = PlannerAgent(client=client)
        plan = await planner.plan(
            goal="Original goal",
            completed_tasks=["Task A"],
            failed_tasks=["Task B"],
            last_test_failures=["AssertionError: expected 1, got 2"],
            cycle_number=2,
        )

        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "Fix test"

        # Verify context was included in the call
        call_args = client.complete.call_args[0][0]
        assert "Already completed tasks" in call_args
        assert "Task A" in call_args
        assert "Previously failed tasks" in call_args
        assert "Task B" in call_args
        assert "Test failures to address" in call_args
        assert "AssertionError" in call_args
        assert "Current cycle number: 2" in call_args


class TestLLMClient:
    def test_run_command_success(self):
        import asyncio
        from furrow.llm import LLMClient

        async def test():
            client = LLMClient()
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = AsyncMock()
                mock_proc.communicate = AsyncMock(return_value=(b"stdout", b"stderr"))
                mock_exec.return_value = mock_proc

                result = await client.run_command("echo", ["hello"])
                assert "stdout" in result
                assert "stderr" in result

        asyncio.run(test())

    def test_run_command_not_found(self):
        import asyncio
        from furrow.llm import LLMClient

        async def test():
            client = LLMClient()
            result = await client.run_command("nonexistent_command_12345")
            assert "Command not found" in result

        asyncio.run(test())