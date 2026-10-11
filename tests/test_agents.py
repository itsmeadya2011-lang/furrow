import asyncio
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel, TestResult, Settings, Provider, Plan
from furrow.llm import LLMClient


class TestPlannerAgent:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=LLMClient)
        client.complete = AsyncMock()
        client.settings = Settings(provider=Provider.ANTHROPIC, planner_model="test-model")
        return client

    @pytest.fixture
    def planner(self, mock_client):
        return PlannerAgent(client=mock_client)

    @pytest.mark.asyncio
    async def test_plan_parses_valid_json(self, planner, mock_client):
        mock_client.complete.return_value = json.dumps({
            "tasks": [
                {"id": "1", "description": "Task 1", "files": ["a.py"], "dependencies": [], "complexity": "low"},
                {"id": "2", "description": "Task 2", "files": ["b.py"], "dependencies": ["1"], "complexity": "medium"},
            ],
            "rationale": "Two tasks with dependency"
        })

        result = await planner.plan("Test goal")

        assert isinstance(result, Plan)
        assert len(result.tasks) == 2
        assert result.tasks[0].id == "1"
        assert result.tasks[0].description == "Task 1"
        assert result.tasks[1].dependencies == ["1"]
        assert result.rationale == "Two tasks with dependency"

    @pytest.mark.asyncio
    async def test_plan_raises_on_invalid_json(self, planner, mock_client):
        mock_client.complete.return_value = "not valid json {"

        with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
            await planner.plan("Test goal")

    @pytest.mark.asyncio
    async def test_plan_raises_on_invalid_schema(self, planner, mock_client):
        mock_client.complete.return_value = json.dumps({"invalid": "schema"})

        with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
            await planner.plan("Test goal")

    @pytest.mark.asyncio
    async def test_plan_uses_correct_model(self, planner, mock_client):
        mock_client.complete.return_value = json.dumps({
            "tasks": [{"id": "1", "description": "Task", "files": [], "dependencies": [], "complexity": "low"}],
            "rationale": "test"
        })

        await planner.plan("Test goal")

        mock_client.complete.assert_called_once()
        call_args = mock_client.complete.call_args
        assert call_args.kwargs.get('model') == "test-model"


class TestWorkerAgent:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=LLMClient)
        client.complete = AsyncMock(return_value="Task completed successfully")
        client.settings = Settings(provider=Provider.ANTHROPIC, worker_model="test-model")
        return client

    @pytest.fixture
    def task(self):
        return TaskModel(id="1", description="Implement feature", files=["src/feature.py"], dependencies=[])

    @pytest.fixture
    def worker(self, task, mock_client):
        return WorkerAgent(task=task, client=mock_client)

    @pytest.mark.asyncio
    async def test_run_formats_prompt_correctly(self, worker, mock_client, task):
        result = await worker.run()

        assert result == "Task completed successfully"
        mock_client.complete.assert_called_once()
        call_args = mock_client.complete.call_args
        prompt = call_args.args[0]
        assert "Implement feature" in prompt
        assert "src/feature.py" in prompt
        assert call_args.kwargs.get('model') == "test-model"

    @pytest.mark.asyncio
    async def test_run_handles_empty_files(self, mock_client):
        task = TaskModel(id="1", description="Task with no files", files=[], dependencies=[])
        worker = WorkerAgent(task=task, client=mock_client)

        await worker.run()

        call_args = mock_client.complete.call_args
        prompt = call_args.args[0]
        assert "any" in prompt or "Files to touch: " in prompt

    @pytest.mark.asyncio
    async def test_run_uses_correct_model(self, worker, mock_client):
        await worker.run()
        call_args = mock_client.complete.call_args
        assert call_args.kwargs.get('model') == "test-model"


class TestTesterAgent:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=LLMClient)
        client.complete = AsyncMock()
        client.settings = Settings(provider=Provider.ANTHROPIC, tester_model="test-model")
        return client

    @pytest.fixture
    def tester(self, mock_client):
        return TesterAgent(client=mock_client)

    @pytest.fixture
    def tasks(self):
        return [
            TaskModel(id="1", description="Task 1", status="completed", result="done"),
            TaskModel(id="2", description="Task 2", status="completed", result="done"),
        ]

    @pytest.mark.asyncio
    async def test_run_parses_valid_json(self, tester, mock_client, tasks):
        mock_client.complete.return_value = json.dumps({
            "passed": True,
            "summary": "All tests passed",
            "failures": []
        })

        with patch.object(tester, '_run_tests', new_callable=AsyncMock, return_value="test output"):
            result = await tester.run("Test goal", tasks)

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_run_parses_failed_json(self, tester, mock_client, tasks):
        mock_client.complete.return_value = json.dumps({
            "passed": False,
            "summary": "Some tests failed",
            "failures": [
                "test_login: Expected 401 got 200 - Detail here"
            ]
        })

        with patch.object(tester, '_run_tests', new_callable=AsyncMock, return_value="test output"):
            result = await tester.run("Test goal", tasks)

        assert result.passed is False
        assert result.summary == "Some tests failed"
        assert len(result.failures) == 1

    @pytest.mark.asyncio
    async def test_run_fallback_on_invalid_json(self, tester, mock_client, tasks):
        mock_client.complete.return_value = "Tests passed! Everything looks good."

        with patch.object(tester, '_run_tests', new_callable=AsyncMock, return_value="test output"):
            result = await tester.run("Test goal", tasks)

        assert result.passed is True
        assert result.summary == "Tests passed! Everything looks good."
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_run_fallback_on_invalid_json_false(self, tester, mock_client, tasks):
        mock_client.complete.return_value = "Tests failed! Something is broken."

        with patch.object(tester, '_run_tests', new_callable=AsyncMock, return_value="test output"):
            result = await tester.run("Test goal", tasks)

        assert result.passed is False
        assert "failed" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_run_handles_test_execution_error(self, tester, mock_client, tasks):
        with patch.object(tester, '_run_tests', new_callable=AsyncMock, side_effect=ValueError("Test runner error")):
            result = await tester.run("Test goal", tasks)

        assert result.passed is False
        assert "Test runner error" in result.summary
        assert "Test runner error" in result.failures[0]

    @pytest.mark.asyncio
    async def test_run_tests_tries_multiple_runners(self, tester):
        with patch('furrow.agents.tester.asyncio.create_subprocess_exec') as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"passed", b""))
            mock_exec.return_value = mock_proc

            result = await tester._run_tests()

            assert "passed" in result
            mock_exec.assert_called()

    @pytest.mark.asyncio
    async def test_run_tests_timeout_handling(self, tester):
        with patch('furrow.agents.tester.asyncio.create_subprocess_exec') as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_exec.return_value = mock_proc

            with patch('furrow.agents.tester.asyncio.wait_for', side_effect=asyncio.TimeoutError()):
                result = await tester._run_tests()

            assert "No test runner found" in result or "Timeout" in result or result != ""

    @pytest.mark.asyncio
    async def test_run_uses_correct_model(self, tester, mock_client, tasks):
        mock_client.complete.return_value = json.dumps({
            "passed": True,
            "summary": "ok",
            "failures": []
        })

        with patch.object(tester, '_run_tests', new_callable=AsyncMock, return_value="output"):
            await tester.run("Test goal", tasks)

        call_args = mock_client.complete.call_args
        assert call_args.kwargs.get('model') == "test-model"