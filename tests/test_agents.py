"""Tests for individual agents (Planner, Worker, Tester)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------


class MockPlannerClient:
    """Mock LLM client for testing PlannerAgent and WorkerAgent."""

    def __init__(self, response: str = ""):
        self._response = response
        self.call_count = 0
        self.calls: list[tuple[str, str]] = []
        self.settings = Settings()

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.call_count += 1
        self.calls.append(("complete", prompt))
        return self._response


class TestPlannerAgent:
    def test_plan_success(self) -> None:
        plan_data = {
            "tasks": [
                {"id": "1", "description": "Task 1", "files": ["a.py"], "dependencies": []},
                {"id": "2", "description": "Task 2", "files": ["b.py"], "dependencies": ["1"]},
            ],
            "rationale": "Build the feature in two steps",
        }
        client = MockPlannerClient(json.dumps(plan_data))
        planner = PlannerAgent(client=client)
        import asyncio

        plan = asyncio.run(planner.plan("Add feature X"))

        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 2
        assert plan.tasks[0].description == "Task 1"
        assert plan.tasks[1].dependencies == ["1"]
        assert plan.rationale == "Build the feature in two steps"

    def test_plan_retries_on_invalid_json(self) -> None:
        """Should retry when LLM returns non-JSON."""
        responses = ["Not JSON at all", json.dumps({"tasks": [], "rationale": "empty"})]

        class CountingClient:
            def __init__(self):
                self.call_count = 0
                self.settings = Settings()

            async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
                resp = responses[self.call_count]
                self.call_count += 1
                return resp

        client = CountingClient()
        planner = PlannerAgent(client=client)
        import asyncio

        plan = asyncio.run(planner.plan("Some goal"))
        assert client.call_count == 2
        assert len(plan.tasks) == 0

    def test_plan_raises_on_persistent_parse_error(self) -> None:
        client = MockPlannerClient("definitely not JSON")
        planner = PlannerAgent(client=client)
        import asyncio

        with pytest.raises(ValueError, match="Failed to parse plan"):
            asyncio.run(planner.plan("Some goal"))
        # Should have tried MAX_RETRIES times
        assert client.call_count == PlannerAgent.MAX_RETRIES

    def test_plan_raises_on_missing_tasks_field(self) -> None:
        client = MockPlannerClient(json.dumps({"rationale": "missing tasks"}))
        planner = PlannerAgent(client=client)
        import asyncio

        with pytest.raises(ValueError):
            asyncio.run(planner.plan("Some goal"))

    def test_empty_plan_returns_empty_tasks(self) -> None:
        client = MockPlannerClient(json.dumps({"tasks": [], "rationale": "done"}))
        planner = PlannerAgent(client=client)
        import asyncio

        plan = asyncio.run(planner.plan("Already done goal"))
        assert len(plan.tasks) == 0


# ---------------------------------------------------------------------------
# WorkerAgent
# ---------------------------------------------------------------------------


class TestWorkerAgent:
    def test_worker_runs_and_returns_result(self) -> None:
        client = MockPlannerClient("Task implementation summary")
        task = TaskModel(id="1", description="Implement feature X", files=["x.py"])
        worker = WorkerAgent(task=task, client=client)
        import asyncio

        result = asyncio.run(worker.run())
        assert result == "Task implementation summary"
        # Should have called complete with worker prompt + task info
        assert len(client.calls) == 1
        prompt = client.calls[0][1]
        assert "Implement feature X" in prompt
        assert "x.py" in prompt

    def test_worker_handles_empty_files(self) -> None:
        client = MockPlannerClient("ok")
        task = TaskModel(id="1", description="Do something", files=[])
        worker = WorkerAgent(task=task, client=client)
        import asyncio

        result = asyncio.run(worker.run())
        assert result == "ok"
        prompt = client.calls[0][1]
        assert "any" in prompt  # Default message for empty files


# ---------------------------------------------------------------------------
# TesterAgent
# ---------------------------------------------------------------------------


class MockTesterClient:
    """Mock LLM client for TesterAgent."""

    def __init__(self, response_data: dict | str):
        self._response = response_data if isinstance(response_data, str) else json.dumps(response_data)
        self.call_count = 0
        self.settings = Settings()

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.call_count += 1
        return self._response


class TestTesterAgent:
    @patch.object(TesterAgent, "_run_tests")
    @pytest.mark.asyncio
    async def test_tester_success(self, mock_run_tests) -> None:
        mock_run_tests.return_value = "3 passed in 0.5s"
        client = MockTesterClient({
            "passed": True,
            "summary": "All tests passed",
            "failures": [],
        })
        tester = TesterAgent(client=client)
        tasks = [TaskModel(id="1", description="Task", status="completed")]

        result = await tester.run("Test goal", tasks)

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    @patch.object(TesterAgent, "_run_tests")
    @pytest.mark.asyncio
    async def test_tester_failure(self, mock_run_tests) -> None:
        mock_run_tests.return_value = "1 failed, 2 passed"
        client = MockTesterClient({
            "passed": False,
            "summary": "1 test failed",
            "failures": ["test_something failed"],
        })
        tester = TesterAgent(client=client)
        tasks = [TaskModel(id="1", description="Task", status="completed")]

        result = await tester.run("Test goal", tasks)

        assert result.passed is False
        assert "test_something failed" in result.failures

    @patch.object(TesterAgent, "_run_tests")
    @pytest.mark.asyncio
    async def test_tester_no_runner_found(self, mock_run_tests) -> None:
        mock_run_tests.return_value = "No test runner found."
        # When no test runner, tester should use LLM evaluation
        client = MockTesterClient({
            "passed": True,
            "summary": "No tests but tasks look good",
            "failures": [],
        })
        tester = TesterAgent(client=client)
        tasks = [TaskModel(id="1", description="Task", status="completed")]

        result = await tester.run("Test goal", tasks)

        assert isinstance(result, TestResult)
        assert result.passed is True

    @patch.object(TesterAgent, "_run_tests")
    @pytest.mark.asyncio
    async def test_tester_fallback_on_invalid_json(self, mock_run_tests) -> None:
        """Should retry with stricter prompt when JSON parsing fails."""
        mock_run_tests.return_value = "3 passed"
        # First response is bad, second is good
        client = MockTesterClient("Not JSON, sorry")
        # Make the mock return good JSON on second call
        original_complete = client.complete

        async def complete(prompt, system="", model=None):
            client.call_count += 1
            if client.call_count == 1:
                return "Not JSON"
            return json.dumps({"passed": True, "summary": "ok", "failures": []})

        client.complete = complete
        tester = TesterAgent(client=client)
        tasks = [TaskModel(id="1", description="Task", status="completed")]

        result = await tester.run("Test goal", tasks)

        assert result.passed is True
        assert client.call_count == 2

    @patch.object(TesterAgent, "_run_tests")
    @pytest.mark.asyncio
    async def test_tester_handles_test_execution_exception(
        self, mock_run_tests
    ) -> None:
        mock_run_tests.side_effect = RuntimeError("Test runner crashed")
        client = MockTesterClient({"passed": True, "summary": "ok", "failures": []})
        tester = TesterAgent(client=client)
        tasks = [TaskModel(id="1", description="Task", status="completed")]

        result = await tester.run("Test goal", tasks)

        assert result.passed is False
        assert "Test runner crashed" in result.summary
        assert "Test execution error" in result.failures[0]

    @patch.object(TesterAgent, "_run_tests")
    @pytest.mark.asyncio
    async def test_tester_retry_with_stricter_prompt(self, mock_run_tests) -> None:
        """When first JSON parse fails, retry with 'IMPORTANT' instruction."""
        mock_run_tests.return_value = "3 passed"

        call_count = [0]

        class CountingClient:
            settings = Settings()

            async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
                call_count[0] += 1
                if call_count[0] == 1:
                    return "Bad response"
                return json.dumps({"passed": True, "summary": "all good", "failures": []})

        tester = TesterAgent(client=CountingClient())  # type: ignore
        tasks = [TaskModel(id="1", description="Task", status="completed")]

        result = await tester.run("Test goal", tasks)
        assert call_count[0] == 2  # First attempt + retry

    @patch.object(TesterAgent, "_run_tests")
    @pytest.mark.asyncio
    async def test_tester_last_resort_heuristic(self, mock_run_tests) -> None:
        """If both JSON parse attempts fail, use heuristic matching."""
        mock_run_tests.return_value = "3 passed"

        class AlwaysBadClient:
            settings = Settings()

            async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
                return "Tests passed successfully"

        tester = TesterAgent(client=AlwaysBadClient())  # type: ignore
        tasks = [TaskModel(id="1", description="Task", status="completed")]

        result = await tester.run("Test goal", tasks)
        assert result.passed is True
        assert "passed" in result.summary.lower()
