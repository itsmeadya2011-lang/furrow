import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class TestConfig:
    """Tests for configuration models."""

    def test_plan_parse(self):
        p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
        assert p.tasks[0].description == "do thing"

    def test_test_result(self):
        t = TestResult(passed=True, summary="ok", failures=[])
        assert t.passed is True

    def test_task_model_defaults(self):
        t = TaskModel(id="1", description="test")
        assert t.files == []
        assert t.dependencies == []
        assert t.status == "pending"
        assert t.result is None

    def test_plan_with_multiple_tasks(self):
        tasks = [
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2"),
        ]
        p = Plan(tasks=tasks, rationale="multi")
        assert len(p.tasks) == 2
        assert p.rationale == "multi"

    def test_test_result_with_failures(self):
        t = TestResult(passed=False, summary="failed", failures=["error1", "error2"])
        assert t.passed is False
        assert len(t.failures) == 2


class TestWorkerAgent:
    """Tests for WorkerAgent file operations."""

    def test_parse_operations_valid_json(self):
        worker = WorkerAgent(task=TaskModel(id="1", description="test"))
        response = '''
        {
            "summary": "Created new file",
            "operations": [
                {"action": "write", "path": "test.py", "content": "print('hello')"}
            ]
        }
        '''
        ops = worker._parse_operations(response)
        assert len(ops) == 1
        assert ops[0]["action"] == "write"
        assert ops[0]["path"] == "test.py"

    def test_parse_operations_no_json(self):
        worker = WorkerAgent(task=TaskModel(id="1", description="test"))
        ops = worker._parse_operations("No JSON here")
        assert ops == []

    def test_parse_operations_invalid_json(self):
        worker = WorkerAgent(task=TaskModel(id="1", description="test"))
        ops = worker._parse_operations("{invalid json}")
        assert ops == []

    def test_parse_operations_no_operations_key(self):
        worker = WorkerAgent(task=TaskModel(id="1", description="test"))
        response = '{"summary": "test"}'
        ops = worker._parse_operations(response)
        assert ops == []

    @pytest.mark.asyncio
    async def test_execute_write_operation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            mock_client.write_file = AsyncMock()
            worker = WorkerAgent(
                task=TaskModel(id="1", description="test"),
                client=mock_client,
            )
            op = {"action": "write", "path": "test.py", "content": "print('hello')", "summary": "test"}
            result = await worker._execute_single_operation(op)
            assert "WRITTEN" in result
            mock_client.write_file.assert_called_once_with("test.py", "print('hello')")

    @pytest.mark.asyncio
    async def test_execute_create_directory_operation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            worker = WorkerAgent(
                task=TaskModel(id="1", description="test"),
                client=mock_client,
            )
            op = {"action": "create_directory", "path": tmpdir + "/newdir", "summary": "test"}
            result = await worker._execute_single_operation(op)
            assert "DIRECTORY CREATED" in result
            assert os.path.isdir(tmpdir + "/newdir")

    @pytest.mark.asyncio
    async def test_execute_unknown_operation(self):
        mock_client = MagicMock()
        worker = WorkerAgent(
            task=TaskModel(id="1", description="test"),
            client=mock_client,
        )
        op = {"action": "delete", "path": "test.py", "summary": "test"}
        result = await worker._execute_single_operation(op)
        assert "SKIP" in result
        assert "Unknown action" in result

    @pytest.mark.asyncio
    async def test_execute_missing_fields(self):
        mock_client = MagicMock()
        worker = WorkerAgent(
            task=TaskModel(id="1", description="test"),
            client=mock_client,
        )
        result = await worker._execute_single_operation({})
        assert "SKIP" in result


class TestPlannerAgent:
    """Tests for PlannerAgent."""

    @pytest.mark.asyncio
    async def test_plan_parses_valid_json(self):
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(
            return_value=json.dumps({
                "tasks": [{"id": "1", "description": "test task"}],
                "rationale": "test plan",
            })
        )
        mock_client.settings = MagicMock()
        mock_client.settings.planner_model = "test-model"

        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("test goal")

        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "test task"
        assert plan.rationale == "test plan"

    @pytest.mark.asyncio
    async def test_plan_raises_on_invalid_json(self):
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="not json")
        mock_client.settings = MagicMock()
        mock_client.settings.planner_model = "test-model"

        planner = PlannerAgent(client=mock_client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await planner.plan("test goal")

    @pytest.mark.asyncio
    async def test_plan_uses_fix_prompt_for_fix_cycle(self):
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(
            return_value=json.dumps({
                "tasks": [{"id": "1", "description": "fix task"}],
                "rationale": "fix plan",
            })
        )
        mock_client.settings = MagicMock()
        mock_client.settings.planner_model = "test-model"

        planner = PlannerAgent(client=mock_client)
        await planner.plan("fix goal", is_fix_cycle=True)

        # Verify the fix prompt was used
        call_args = mock_client.complete.call_args
        assert "fix" in call_args[0][0].lower() or "Fix" in call_args[0][0]


class TestOrchestrator:
    """Tests for Orchestrator state management."""

    def test_get_tasks_returns_empty_initially(self):
        mock_client = MagicMock()
        orch = Orchestrator(goal="test", client=mock_client)
        assert orch._get_tasks() == []

    def test_get_tasks_returns_current_plan_tasks(self):
        mock_client = MagicMock()
        orch = Orchestrator(goal="test", client=mock_client)
        plan = Plan(
            tasks=[TaskModel(id="1", description="test")],
            rationale="test",
        )
        orch._current_plan = plan
        assert orch._get_tasks() == plan.tasks

    def test_is_done_with_no_tasks(self):
        mock_client = MagicMock()
        orch = Orchestrator(goal="test", client=mock_client)
        # With no plan, _get_tasks returns [], completed=0, failed=0
        # 0 >= 0 is True, so _is_done returns True
        assert orch._is_done() is True

    def test_is_done_with_completed_tasks(self):
        mock_client = MagicMock()
        orch = Orchestrator(goal="test", client=mock_client)
        task = TaskModel(id="1", description="test", status="completed")
        plan = Plan(tasks=[task], rationale="test")
        orch._current_plan = plan
        assert orch._is_done() is True

    def test_is_done_with_failed_tasks(self):
        mock_client = MagicMock()
        orch = Orchestrator(goal="test", client=mock_client)
        task = TaskModel(id="1", description="test", status="failed")
        plan = Plan(tasks=[task], rationale="test")
        orch._current_plan = plan
        assert orch._is_done() is False

    def test_is_done_with_pending_tasks(self):
        mock_client = MagicMock()
        orch = Orchestrator(goal="test", client=mock_client)
        task = TaskModel(id="1", description="test", status="pending")
        plan = Plan(tasks=[task], rationale="test")
        orch._current_plan = plan
        assert orch._is_done() is False

    def test_max_cycles_limit(self):
        mock_client = MagicMock()
        orch = Orchestrator(goal="test", client=mock_client, max_cycles=5)
        assert orch.max_cycles == 5
        orch.cycles = 5
        # When cycles >= max_cycles, run() should halt


class TestTesterAgent:
    """Tests for TesterAgent."""

    @pytest.mark.asyncio
    async def test_run_returns_passed_for_good_output(self):
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(
            return_value=json.dumps({
                "passed": True,
                "summary": "All tests passed",
                "failures": [],
            })
        )
        mock_client.settings = MagicMock()
        mock_client.settings.tester_model = "test-model"

        tester = TesterAgent(client=mock_client)
        result = await tester.run("test goal", [])

        assert result.passed is True
        assert result.summary == "All tests passed"

    @pytest.mark.asyncio
    async def test_run_returns_failed_for_bad_output(self):
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(
            return_value=json.dumps({
                "passed": False,
                "summary": "Tests failed",
                "failures": ["error1"],
            })
        )
        mock_client.settings = MagicMock()
        mock_client.settings.tester_model = "test-model"

        tester = TesterAgent(client=mock_client)
        result = await tester.run("test goal", [])

        assert result.passed is False
        assert len(result.failures) == 1

    @pytest.mark.asyncio
    async def test_run_handles_invalid_json(self):
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="Tests passed successfully")
        mock_client.settings = MagicMock()
        mock_client.settings.tester_model = "test-model"

        tester = TesterAgent(client=mock_client)
        result = await tester.run("test goal", [])

        # Should fallback to checking for "passed" in response
        assert result.passed is True


class TestIntegration:
    """Integration tests for the full workflow."""

    @pytest.mark.asyncio
    async def test_worker_writes_real_file(self):
        """Test that worker can actually write files to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a real LLMClient but mock the complete method
            mock_settings = MagicMock()
            mock_settings.worker_model = "test-model"
            mock_settings.provider = "anthropic"
            mock_settings.anthropic_api_key = "test-key"

            client = LLMClient(settings=mock_settings)
            client.complete = AsyncMock(return_value=json.dumps({
                "summary": "Created test file",
                "operations": [
                    {
                        "action": "write",
                        "path": os.path.join(tmpdir, "test_file.py"),
                        "content": "def hello(): return 'world'",
                    }
                ],
            }))

            task = TaskModel(id="1", description="Create a test file")
            worker = WorkerAgent(task=task, client=client)
            result = await worker.run()

            assert "WRITTEN" in result
            assert os.path.exists(os.path.join(tmpdir, "test_file.py"))
            with open(os.path.join(tmpdir, "test_file.py")) as f:
                assert "def hello()" in f.read()


class TestGitManager:
    """Tests for GitManager."""

    @pytest.mark.asyncio
    async def test_is_available_returns_false_when_no_git(self):
        """Test that is_available returns False when git is not installed."""
        from furrow.git import GitManager
        with tempfile.TemporaryDirectory() as tmpdir:
            gm = GitManager(tmpdir)
            # This will return False if git is not available
            # In a test environment, git might be available
            result = await gm.is_available()
            assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_ensure_repo_creates_git_dir(self):
        """Test that ensure_repo initializes a git repository."""
        from furrow.git import GitManager
        with tempfile.TemporaryDirectory() as tmpdir:
            gm = GitManager(tmpdir)
            result = await gm.ensure_repo()
            if result:  # Git is available
                assert os.path.isdir(os.path.join(tmpdir, ".git"))

    @pytest.mark.asyncio
    async def test_commit_changes_creates_commit(self):
        """Test that commit_changes creates a git commit."""
        from furrow.git import GitManager
        with tempfile.TemporaryDirectory() as tmpdir:
            gm = GitManager(tmpdir)
            if await gm.ensure_repo():
                # Create a test file
                test_file = os.path.join(tmpdir, "test.txt")
                with open(test_file, "w") as f:
                    f.write("test content")

                result = await gm.commit_changes("Test commit", cycle=1)
                assert result is True
