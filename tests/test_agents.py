import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult


class TestPlannerAgent:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.settings = Settings(planner_model="test-model")
        client.complete = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_plan_valid_json(self, mock_client):
        plan_data = {
            "tasks": [
                {"id": "1", "description": "Task 1", "files": ["a.py"], "dependencies": []},
                {"id": "2", "description": "Task 2", "files": ["b.py"], "dependencies": ["1"]},
            ],
            "rationale": "Test rationale",
        }
        mock_client.complete.return_value = json.dumps(plan_data)

        agent = PlannerAgent(client=mock_client)
        plan = await agent.plan("test goal")

        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 2
        assert plan.tasks[0].id == "1"
        assert plan.tasks[1].dependencies == ["1"]
        assert plan.rationale == "Test rationale"

    @pytest.mark.asyncio
    async def test_plan_malformed_json_retry_succeeds(self, mock_client):
        plan_data = {
            "tasks": [{"id": "1", "description": "Task 1", "files": [], "dependencies": []}],
            "rationale": "Retry worked",
        }
        mock_client.complete.side_effect = [
            "not valid json {{{",
            json.dumps(plan_data),
        ]

        agent = PlannerAgent(client=mock_client)
        plan = await agent.plan("test goal")

        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 1
        assert plan.rationale == "Retry worked"
        assert mock_client.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_plan_malformed_json_retry_fails(self, mock_client):
        mock_client.complete.side_effect = [
            "not valid json",
            "still not valid json",
        ]

        agent = PlannerAgent(client=mock_client)
        with pytest.raises(ValueError, match="Failed to parse plan from LLM after retry"):
            await agent.plan("test goal")

    @pytest.mark.asyncio
    async def test_plan_with_context(self, mock_client):
        plan_data = {
            "tasks": [{"id": "1", "description": "Fix task", "files": [], "dependencies": []}],
            "rationale": "Context aware",
        }
        mock_client.complete.return_value = json.dumps(plan_data)

        agent = PlannerAgent(client=mock_client)
        context = {"failed_tasks": ["old task"], "test_results": ["error: something failed"]}
        plan = await agent.plan("test goal", context=context)

        assert isinstance(plan, Plan)
        # Verify the prompt includes context
        call_args = mock_client.complete.call_args
        prompt = call_args[0][0]
        assert "Previously failed tasks:" in prompt
        assert "old task" in prompt
        assert "Previous test results:" in prompt
        assert "error: something failed" in prompt

    @pytest.mark.asyncio
    async def test_plan_empty_context(self, mock_client):
        plan_data = {"tasks": [], "rationale": "No tasks needed"}
        mock_client.complete.return_value = json.dumps(plan_data)

        agent = PlannerAgent(client=mock_client)
        plan = await agent.plan("test goal", context={})

        assert isinstance(plan, Plan)
        assert plan.tasks == []

    @pytest.mark.asyncio
    async def test_plan_none_context(self, mock_client):
        plan_data = {"tasks": [], "rationale": "No tasks needed"}
        mock_client.complete.return_value = json.dumps(plan_data)

        agent = PlannerAgent(client=mock_client)
        plan = await agent.plan("test goal", context=None)

        assert isinstance(plan, Plan)


class TestWorkerAgent:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.settings = Settings(worker_model="test-model")
        client.complete = AsyncMock(return_value="Task completed successfully")
        client.read_file = AsyncMock()
        return client

    @pytest.fixture
    def task(self):
        return TaskModel(id="1", description="Test task", files=["src/main.py"])

    @pytest.mark.asyncio
    async def test_run_reads_files(self, mock_client, task, tmp_path):
        test_file = tmp_path / "src" / "main.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("print('hello')")
        mock_client.read_file.return_value = "print('hello')"

        agent = WorkerAgent(task=task, client=mock_client)
        result = await agent.run()

        assert result == "Task completed successfully"
        mock_client.read_file.assert_called_once()
        mock_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_handles_missing_file(self, mock_client, task):
        mock_client.read_file.side_effect = FileNotFoundError("not found")

        agent = WorkerAgent(task=task, client=mock_client)
        result = await agent.run()

        assert result == "Task completed successfully"
        mock_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_read_files_multiple(self, mock_client):
        task = TaskModel(id="1", description="Test", files=["a.py", "b.py"])
        mock_client.read_file.side_effect = ["content a", "content b"]

        agent = WorkerAgent(task=task, client=mock_client)
        files = await agent._read_files()

        assert files == {"a.py": "content a", "b.py": "content b"}
        assert mock_client.read_file.call_count == 2

    @pytest.mark.asyncio
    async def test_read_files_skips_missing(self, mock_client):
        task = TaskModel(id="1", description="Test", files=["exists.py", "missing.py"])
        mock_client.read_file.side_effect = ["content", FileNotFoundError("not found")]

        agent = WorkerAgent(task=task, client=mock_client)
        files = await agent._read_files()

        assert files == {"exists.py": "content"}

    @pytest.mark.asyncio
    async def test_build_prompt_includes_files_and_deps(self, mock_client):
        task = TaskModel(
            id="1",
            description="Implement feature",
            files=["src/feature.py"],
            dependencies=["0"],
        )
        mock_client.read_file.return_value = "existing code"

        agent = WorkerAgent(task=task, client=mock_client)
        prompt = agent._build_prompt({"src/feature.py": "existing code"})

        assert "Implement feature" in prompt
        assert "src/feature.py" in prompt
        assert "Depends on: 0" in prompt
        assert "existing code" in prompt

    @pytest.mark.asyncio
    async def test_build_prompt_no_files(self, mock_client):
        task = TaskModel(id="1", description="Simple task")

        agent = WorkerAgent(task=task, client=mock_client)
        prompt = agent._build_prompt({})

        assert "Simple task" in prompt
        assert "Files to touch" not in prompt
        assert "Depends on" not in prompt


class TestTesterAgent:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.settings = Settings(tester_model="test-model", test_commands=[])
        client.complete = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_get_test_commands_from_env(self, mock_client, monkeypatch):
        monkeypatch.setenv("FURROW_TEST_COMMANDS", "pytest -v, cargo test")
        agent = TesterAgent(client=mock_client)
        commands = agent._get_test_commands()

        assert commands == [["pytest", "-v"], ["cargo", "test"]]

    @pytest.mark.asyncio
    async def test_get_test_commands_from_settings(self, mock_client):
        mock_client.settings.test_commands = ["npm test", "python -m pytest"]
        agent = TesterAgent(client=mock_client)
        commands = agent._get_test_commands()

        assert commands == [["npm", "test"], ["python", "-m", "pytest"]]

    @pytest.mark.asyncio
    async def test_get_test_commands_defaults(self, mock_client, monkeypatch):
        monkeypatch.delenv("FURROW_TEST_COMMANDS", raising=False)
        mock_client.settings.test_commands = []
        agent = TesterAgent(client=mock_client)
        commands = agent._get_test_commands()

        assert commands[0] == ["pytest", "-q"]
        assert ["cargo", "test", "-q"] in commands
        assert ["go", "test", "./..."] in commands

    def test_truncate_short_text(self, mock_client):
        agent = TesterAgent(client=mock_client)
        text = "short"
        result = agent._truncate(text)
        assert result == "short"

    def test_truncate_long_text(self, mock_client):
        agent = TesterAgent(client=mock_client)
        text = "x" * 10000
        result = agent._truncate(text)
        assert len(result) <= 8192 + 100  # MAX_OUTPUT_BYTES + truncation message
        assert "[truncated" in result
        assert text[:4096] in result
        assert text[-4096:] in result

    def test_combine_output_both(self, mock_client):
        agent = TesterAgent(client=mock_client)
        result = agent._combine_output("stdout content", "stderr content")
        assert "stdout:\nstdout content" in result
        assert "stderr:\nstderr content" in result

    def test_combine_output_stdout_only(self, mock_client):
        agent = TesterAgent(client=mock_client)
        result = agent._combine_output("stdout only", "")
        assert result == "stdout:\nstdout only"

    def test_combine_output_stderr_only(self, mock_client):
        agent = TesterAgent(client=mock_client)
        result = agent._combine_output("", "stderr only")
        assert result == "stderr:\nstderr only"

    def test_combine_output_both_empty(self, mock_client):
        agent = TesterAgent(client=mock_client)
        result = agent._combine_output("", "")
        assert result == ""

    @pytest.mark.asyncio
    async def test_run_tests_passed(self, mock_client):
        test_result_data = {"passed": True, "summary": "All tests passed", "failures": []}
        mock_client.complete.return_value = json.dumps(test_result_data)

        with patch.object(TesterAgent, "_run_tests", AsyncMock(return_value="test output")):
            agent = TesterAgent(client=mock_client)
            result = await agent.run("test goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_run_tests_failed(self, mock_client):
        test_result_data = {
            "passed": False,
            "summary": "Tests failed",
            "failures": ["test_foo.py:10: AssertionError"],
        }
        mock_client.complete.return_value = json.dumps(test_result_data)

        with patch.object(TesterAgent, "_run_tests", AsyncMock(return_value="test output")):
            agent = TesterAgent(client=mock_client)
            result = await agent.run("test goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is False
        assert len(result.failures) == 1

    @pytest.mark.asyncio
    async def test_run_malformed_json_fallback(self, mock_client):
        mock_client.complete.return_value = "not json at all but passed somewhere"

        with patch.object(TesterAgent, "_run_tests", AsyncMock(return_value="test output")):
            agent = TesterAgent(client=mock_client)
            result = await agent.run("test goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is True  # "passed" in response.lower()

    @pytest.mark.asyncio
    async def test_run_malformed_json_fallback_failed(self, mock_client):
        mock_client.complete.return_value = "not json and failed"

        with patch.object(TesterAgent, "_run_tests", AsyncMock(return_value="test output")):
            agent = TesterAgent(client=mock_client)
            result = await agent.run("test goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_run_exception_during_tests(self, mock_client):
        with patch.object(TesterAgent, "_run_tests", AsyncMock(side_effect=Exception("test error"))):
            agent = TesterAgent(client=mock_client)
            result = await agent.run("test goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is False
        assert "test error" in result.summary
        assert result.failures == ["test error"]


class TestAgentIntegration:
    @pytest.mark.asyncio
    async def test_planner_worker_tester_chain(self):
        settings = Settings(
            planner_model="test",
            worker_model="test",
            tester_model="test",
        )

        mock_client = MagicMock()
        mock_client.settings = settings
        mock_client.complete = AsyncMock()
        mock_client.read_file = AsyncMock(return_value="")

        # Planner returns a plan
        plan_data = {
            "tasks": [
                {"id": "1", "description": "Task 1", "files": ["a.py"], "dependencies": []}
            ],
            "rationale": "Test",
        }
        mock_client.complete.side_effect = [
            json.dumps(plan_data),  # planner
            "Worker done",  # worker
            json.dumps({"passed": True, "summary": "ok", "failures": []}),  # tester
        ]

        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("goal")
        assert len(plan.tasks) == 1

        worker = WorkerAgent(task=plan.tasks[0], client=mock_client)
        worker_result = await worker.run()
        assert worker_result == "Worker done"

        tester = TesterAgent(client=mock_client)
        with patch.object(TesterAgent, "_run_tests", AsyncMock(return_value="output")):
            test_result = await tester.run("goal", plan.tasks)
        assert test_result.passed is True