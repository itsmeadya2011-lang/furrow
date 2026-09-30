import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


class MockLLMClient:
    """Mock LLM client that returns canned responses for testing."""

    def __init__(self, response: str = "{}", responses: list | None = None):
        if responses is not None:
            self._responses = list(responses)
        else:
            self._responses = [response]
        self._response_index = 0
        self.settings = Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
        )
        self.call_count = 0
        self.calls: list[dict] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.call_count += 1
        self.calls.append({"prompt": prompt, "system": system, "model": model})
        if self._response_index < len(self._responses):
            response = self._responses[self._response_index]
        elif self._responses:
            response = self._responses[-1]
        else:
            response = "{}"
        self._response_index += 1
        return response

    async def read_file(self, path: str | Path) -> str:
        return ""

    async def write_file(self, path: str | Path, content: str) -> None:
        pass

    def list_files(self, directory: str | Path) -> list[str]:
        return []


class TestPlannerAgent:
    async def test_plan_returns_valid_plan(self):
        plan_json = json.dumps(
            {
                "tasks": [
                    {
                        "id": "1",
                        "description": "create file",
                        "files": ["file.py"],
                        "dependencies": [],
                    },
                    {
                        "id": "2",
                        "description": "test file",
                        "files": ["test_file.py"],
                        "dependencies": ["1"],
                    },
                ],
                "rationale": "Break down the task",
            }
        )
        client = MockLLMClient(response=plan_json)
        planner = PlannerAgent(client=client)
        plan = await planner.plan("create and test a file")
        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 2
        assert plan.tasks[0].id == "1"
        assert plan.tasks[0].description == "create file"
        assert plan.tasks[0].files == ["file.py"]
        assert plan.tasks[1].dependencies == ["1"]
        assert plan.rationale == "Break down the task"
        assert client.call_count == 1

    async def test_plan_single_task(self):
        plan_json = json.dumps(
            {
                "tasks": [
                    {"id": "1", "description": "do one thing", "files": [], "dependencies": []}
                ],
                "rationale": "simple task",
            }
        )
        client = MockLLMClient(response=plan_json)
        planner = PlannerAgent(client=client)
        plan = await planner.plan("do something")
        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "1"

    async def test_plan_raises_on_invalid_json(self):
        client = MockLLMClient(response="this is not valid json {{{")
        planner = PlannerAgent(client=client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await planner.plan("some goal")

    async def test_plan_raises_on_missing_tasks_field(self):
        plan_json = json.dumps({"rationale": "missing tasks"})
        client = MockLLMClient(response=plan_json)
        planner = PlannerAgent(client=client)
        with pytest.raises(ValueError):
            await planner.plan("some goal")

    async def test_plan_uses_planner_model(self):
        plan_json = json.dumps({"tasks": [], "rationale": "done"})
        client = MockLLMClient(response=plan_json)
        planner = PlannerAgent(client=client)
        await planner.plan("some goal")
        assert client.calls[0]["model"] == client.settings.planner_model


class TestWorkerAgent:
    async def test_run_returns_string_summary(self):
        summary = "Implemented feature X in src/foo.py"
        client = MockLLMClient(response=summary)
        task = TaskModel(id="1", description="implement feature X", files=["src/foo.py"])
        worker = WorkerAgent(task=task, client=client)
        result = await worker.run()
        assert isinstance(result, str)
        assert result == summary
        assert client.call_count == 1

    async def test_run_uses_worker_model(self):
        client = MockLLMClient(response="done")
        task = TaskModel(id="1", description="task", files=[])
        worker = WorkerAgent(task=task, client=client)
        await worker.run()
        assert client.calls[0]["model"] == client.settings.worker_model

    async def test_run_includes_task_description_in_prompt(self):
        client = MockLLMClient(response="done")
        task = TaskModel(id="1", description="build a castle", files=["castle.py"])
        worker = WorkerAgent(task=task, client=client)
        await worker.run()
        assert "build a castle" in client.calls[0]["prompt"]

    async def test_run_includes_files_in_prompt(self):
        client = MockLLMClient(response="done")
        task = TaskModel(id="1", description="task", files=["a.py", "b.py"])
        worker = WorkerAgent(task=task, client=client)
        await worker.run()
        assert "a.py" in client.calls[0]["prompt"]
        assert "b.py" in client.calls[0]["prompt"]

    async def test_run_no_files_uses_any(self):
        client = MockLLMClient(response="done")
        task = TaskModel(id="1", description="task", files=[])
        worker = WorkerAgent(task=task, client=client)
        await worker.run()
        assert "any" in client.calls[0]["prompt"]


class TestTesterAgent:
    async def test_run_returns_passed_test_result(self):
        client = MockLLMClient(
            response=json.dumps(
                {
                    "passed": True,
                    "summary": "All 10 tests passed",
                    "failures": [],
                }
            )
        )
        tester = TesterAgent(client=client)
        tester._run_tests = AsyncMock(return_value="test output: all passed")
        result = await tester.run(
            "some goal",
            [
                TaskModel(id="1", description="task1", status="completed"),
            ],
        )
        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "All 10 tests passed"
        assert result.failures == []
        assert client.call_count == 1

    async def test_run_returns_passed_with_failures(self):
        client = MockLLMClient(
            response=json.dumps(
                {
                    "passed": False,
                    "summary": "2 tests failed",
                    "failures": ["test_a: assertion error", "test_b: timeout"],
                }
            )
        )
        tester = TesterAgent(client=client)
        tester._run_tests = AsyncMock(return_value="test output: 2 failed")
        result = await tester.run(
            "some goal",
            [
                TaskModel(id="1", description="task1", status="completed"),
            ],
        )
        assert result.passed is False
        assert result.summary == "2 tests failed"
        assert len(result.failures) == 2
        assert "test_a" in result.failures[0]

    async def test_run_returns_failed_on_invalid_json(self):
        client = MockLLMClient(response="not valid json at all")
        tester = TesterAgent(client=client)
        tester._run_tests = AsyncMock(return_value="test output")
        result = await tester.run(
            "some goal",
            [
                TaskModel(id="1", description="task1", status="completed"),
            ],
        )
        assert result.passed is False
        assert result.summary == "not valid json at all"
        assert result.failures == []

    async def test_run_returns_failed_when_run_tests_exception(self):
        client = MockLLMClient(
            response=json.dumps({"passed": True, "summary": "ok", "failures": []})
        )
        tester = TesterAgent(client=client)
        tester._run_tests = AsyncMock(side_effect=OSError("no test runner"))
        result = await tester.run("some goal", [])
        assert result.passed is False
        assert "no test runner" in result.summary
        assert client.call_count == 0

    async def test_run_uses_tester_model(self):
        client = MockLLMClient(
            response=json.dumps({"passed": True, "summary": "ok", "failures": []})
        )
        tester = TesterAgent(client=client)
        tester._run_tests = AsyncMock(return_value="test output")
        await tester.run("goal", [])
        assert client.calls[0]["model"] == client.settings.tester_model

    async def test_run_no_test_runner_found(self):
        client = MockLLMClient(
            response=json.dumps(
                {
                    "passed": False,
                    "summary": "No tests found",
                    "failures": ["No test runner found"],
                }
            )
        )
        tester = TesterAgent(client=client)
        tester._run_tests = AsyncMock(return_value="No test runner found.")
        result = await tester.run("some goal", [])
        assert result.passed is False
        assert "No tests found" in result.summary


class TestTesterRunTests:
    async def test_run_tests_no_runner_found(self):
        tester = TesterAgent(client=MockLLMClient())
        with patch(
            "asyncio.create_subprocess_exec", side_effect=FileNotFoundError("command not found")
        ):
            result = await tester._run_tests()
            assert result == "No test runner found."

    async def test_run_tests_timeout_then_no_runner(self):
        tester = TesterAgent(client=MockLLMClient())

        call_count = 0

        async def mock_exec(*cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                proc = AsyncMock()
                proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
                proc.kill = MagicMock()
                return proc
            raise FileNotFoundError("not found")

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            result = await tester._run_tests()
            assert result == "No test runner found."

    async def test_run_tests_returns_output(self):
        tester = TesterAgent(client=MockLLMClient())
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"test stdout", b"test stderr"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await tester._run_tests()
            assert "test stdout" in result
            assert "test stderr" in result
