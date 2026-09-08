"""Tests for the Furrow agents (planner, worker, tester)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient


@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    client.settings = MagicMock()
    client.settings.planner_model = "test-planner"
    client.settings.worker_model = "test-worker"
    client.settings.tester_model = "test-tester"
    client.settings.workspace = Settings().workspace
    client.complete = AsyncMock(return_value="{}")
    client.complete_with_tools = AsyncMock(return_value="done")
    client.close = AsyncMock()
    return client


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_parses_valid_json(self, mock_client):
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "tasks": [{"id": "1", "description": "task A", "files": ["a.py"], "dependencies": []}],
            "rationale": "test",
        }))
        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("test goal")
        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "task A"

    @pytest.mark.asyncio
    async def test_plan_parses_json_from_markdown(self, mock_client):
        mock_client.complete = AsyncMock(return_value="```json\n" + json.dumps({
            "tasks": [{"id": "1", "description": "task A", "files": [], "dependencies": []}],
            "rationale": "test",
        }) + "\n```")
        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("test goal")
        assert len(plan.tasks) == 1

    @pytest.mark.asyncio
    async def test_plan_raises_on_invalid_response(self, mock_client):
        mock_client.complete = AsyncMock(return_value="not json at all")
        planner = PlannerAgent(client=mock_client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await planner.plan("test goal")

    @pytest.mark.asyncio
    async def test_plan_retry_on_failure(self, mock_client):
        """Planner should retry LLM calls that raise exceptions."""
        call_count = 0

        async def mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("API error")
            return json.dumps({
                "tasks": [{"id": "1", "description": "task", "files": [], "dependencies": []}],
                "rationale": "ok",
            })

        mock_client.complete = mock_complete
        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("test goal")
        assert len(plan.tasks) == 1

    def test_extract_json_block_from_markdown(self):
        text = '```json\n{"key": "value"}\n```'
        result = PlannerAgent._extract_json_block(text)
        assert result == '{"key": "value"}'

    def test_extract_json_block_raw(self):
        text = 'some text {"key": "value"} more text'
        result = PlannerAgent._extract_json_block(text)
        assert result == '{"key": "value"}'

    def test_extract_json_block_none(self):
        text = "no json here"
        result = PlannerAgent._extract_json_block(text)
        assert result is None


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_worker_uses_tool_loop(self, mock_client):
        mock_client.complete_with_tools = AsyncMock(return_value="Task completed successfully")
        task = TaskModel(id="1", description="fix bug", files=["bug.py"])
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()
        assert result == "Task completed successfully"
        mock_client.complete_with_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_worker_passes_task_info(self, mock_client):
        mock_client.complete_with_tools = AsyncMock(return_value="done")
        task = TaskModel(id="42", description="implement feature", files=["a.py", "b.py"], dependencies=["1"])
        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()

        call_args = mock_client.complete_with_tools.call_args
        messages = call_args.kwargs["messages"]
        # The first message should contain the task description
        prompt = messages[0]["content"]
        assert "implement feature" in prompt
        assert "42" in prompt  # task id


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_tester_passes_on_success(self, mock_client):
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "passed": True,
            "summary": "All tests passed",
            "failures": [],
        }))
        tester = TesterAgent(client=mock_client)
        with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="test output"):
            result = await tester.run("test goal", [])
        assert result.passed is True
        assert result.summary == "All tests passed"

    @pytest.mark.asyncio
    async def test_tester_fails_on_failure(self, mock_client):
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "passed": False,
            "summary": "2 tests failed",
            "failures": ["test_a failed", "test_b failed"],
        }))
        tester = TesterAgent(client=mock_client)
        with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="test output"):
            with patch.object(WorkerAgent, "run", new_callable=AsyncMock, return_value="fixed"):
                result = await tester.run("test goal", [])
        assert result.passed is False
        assert len(result.failures) == 2

    @pytest.mark.asyncio
    async def test_tester_fallback_on_parse_error(self, mock_client):
        mock_client.complete = AsyncMock(return_value="all tests passed")
        tester = TesterAgent(client=mock_client)
        with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="test output"):
            result = await tester.run("test goal", [])
        assert result.passed is True

    def test_detect_test_commands_with_pytest_ini(self, tmp_path):
        (tmp_path / "pytest.ini").write_text("[pytest]")
        settings = Settings(workspace=tmp_path)
        client = LLMClient(settings=settings)
        tester = TesterAgent(client=client)
        commands = tester._detect_test_commands(tmp_path)
        assert ["pytest", "-q"] in commands

    def test_detect_test_commands_with_package_json(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "test"}')
        settings = Settings(workspace=tmp_path)
        client = LLMClient(settings=settings)
        tester = TesterAgent(client=client)
        commands = tester._detect_test_commands(tmp_path)
        assert ["npm", "test", "--", "--silent"] in commands

    def test_detect_test_commands_fallback(self, tmp_path):
        settings = Settings(workspace=tmp_path)
        client = LLMClient(settings=settings)
        tester = TesterAgent(client=client)
        commands = tester._detect_test_commands(tmp_path)
        assert ["python", "-m", "pytest", "-q"] in commands

    @pytest.mark.asyncio
    async def test_fix_failures_spawns_worker(self, tmp_path):
        settings = Settings(workspace=tmp_path)
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = settings
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "passed": False,
            "summary": "fail",
            "failures": ["test failed"],
        }))
        mock_client.complete_with_tools = AsyncMock(return_value="fixed")
        tester = TesterAgent(client=mock_client)
        with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="output"):
            with patch.object(WorkerAgent, "run", new_callable=AsyncMock, return_value="fixed") as mock_worker:
                result = await tester.run("fix the bug", [])
        mock_worker.assert_called_once()
        assert result is not None
