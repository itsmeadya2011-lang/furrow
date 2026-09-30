"""Tests for the Agent classes (Planner, Worker, Tester)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_parses_json_response(self):
        """Planner should parse a valid JSON plan from the LLM."""
        client = MagicMock(spec=LLMClient)
        client.settings = MagicMock()
        client.settings.planner_model = "test-model"
        valid_response = json.dumps(
            {
                "tasks": [{"id": "1", "description": "do thing", "files": [], "dependencies": []}],
                "rationale": "test",
            }
        )
        client.complete = AsyncMock(return_value=valid_response)

        planner = PlannerAgent(client=client)
        plan = await planner.plan("build a thing")

        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "do thing"
        assert plan.rationale == "test"

    @pytest.mark.asyncio
    async def test_plan_invalid_json_raises(self):
        """Planner should raise ValueError on invalid JSON."""
        client = MagicMock(spec=LLMClient)
        client.settings = MagicMock()
        client.settings.planner_model = "test-model"
        client.complete = AsyncMock(return_value="not json at all")

        planner = PlannerAgent(client=client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await planner.plan("do something")

    @pytest.mark.asyncio
    async def test_plan_with_multiple_tasks(self):
        client = MagicMock(spec=LLMClient)
        client.settings = MagicMock()
        client.settings.planner_model = "test-model"
        valid_response = json.dumps(
            {
                "tasks": [
                    {"id": "1", "description": "task one", "files": ["a.py"], "dependencies": []},
                    {
                        "id": "2",
                        "description": "task two",
                        "files": ["b.py"],
                        "dependencies": ["1"],
                    },
                ],
                "rationale": "multi task",
            }
        )
        client.complete = AsyncMock(return_value=valid_response)

        planner = PlannerAgent(client=client)
        plan = await planner.plan("multi goal")

        assert len(plan.tasks) == 2
        assert plan.tasks[1].dependencies == ["1"]


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_worker_returns_llm_response(self):
        """Worker should return the LLM's text response."""
        client = MagicMock(spec=LLMClient)
        client.settings = MagicMock()
        client.settings.worker_model = "test-model"
        client.complete = AsyncMock(return_value="I changed the code!")

        task = TaskModel(id="1", description="add a function", files=["main.py"])
        worker = WorkerAgent(task=task, client=client)
        result = await worker.run()

        assert result == "I changed the code!"
        client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_worker_includes_task_info_in_prompt(self):
        """Worker prompt should include task description and files."""
        client = MagicMock(spec=LLMClient)
        client.settings = MagicMock()
        client.settings.worker_model = "test-model"
        client.complete = AsyncMock(return_value="done")

        task = TaskModel(id="3", description="build API endpoint", files=["api.py", "models.py"])
        worker = WorkerAgent(task=task, client=client)
        await worker.run()

        prompt = client.complete.call_args[0][0]
        assert "build API endpoint" in prompt
        assert "api.py" in prompt
        assert "models.py" in prompt

    @pytest.mark.asyncio
    async def test_worker_no_files_hint(self):
        """Worker should handle empty files list gracefully."""
        client = MagicMock(spec=LLMClient)
        client.settings = MagicMock()
        client.settings.worker_model = "test-model"
        client.complete = AsyncMock(return_value="done")

        task = TaskModel(id="1", description="do something")
        worker = WorkerAgent(task=task, client=client)
        await worker.run()

        prompt = client.complete.call_args[0][0]
        assert "any" in prompt


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_tester_parses_passing_result(self):
        """Tester should parse a passing TestResult from the LLM."""
        client = MagicMock(spec=LLMClient)
        client.settings = MagicMock()
        client.settings.tester_model = "test-model"
        client.settings.workspace = MagicMock()
        client.settings.workspace.exists = MagicMock(return_value=False)
        client.complete = AsyncMock(
            return_value=json.dumps(
                {
                    "passed": True,
                    "summary": "All tests passed",
                    "failures": [],
                }
            )
        )

        tester = TesterAgent(client=client)
        with patch.object(tester, "_detect_test_commands", return_value=[]):
            result = await tester.run("build a thing", [])

        assert result.passed is True
        assert result.summary == "All tests passed"

    @pytest.mark.asyncio
    async def test_tester_parses_failing_result(self):
        """Tester should parse a failing TestResult from the LLM."""
        client = MagicMock(spec=LLMClient)
        client.settings = MagicMock()
        client.settings.tester_model = "test-model"
        client.settings.workspace = MagicMock()
        client.settings.workspace.exists = MagicMock(return_value=False)
        client.complete = AsyncMock(
            return_value=json.dumps(
                {
                    "passed": False,
                    "summary": "2 tests failed",
                    "failures": ["test_a failed", "test_b failed"],
                }
            )
        )

        tester = TesterAgent(client=client)
        with patch.object(tester, "_detect_test_commands", return_value=[]):
            result = await tester.run("build a thing", [])

        assert result.passed is False
        assert len(result.failures) == 2

    @pytest.mark.asyncio
    async def test_tester_fallback_on_invalid_json(self):
        """Tester should fall back to keyword matching on invalid JSON."""
        client = MagicMock(spec=LLMClient)
        client.settings = MagicMock()
        client.settings.tester_model = "test-model"
        client.settings.workspace = MagicMock()
        client.settings.workspace.exists = MagicMock(return_value=False)
        client.complete = AsyncMock(return_value="The tests passed successfully")

        tester = TesterAgent(client=client)
        with patch.object(tester, "_detect_test_commands", return_value=[]):
            result = await tester.run("build a thing", [])

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_tester_no_runner_found(self):
        """Tester should return a no-runner message when no test runner is available."""
        client = MagicMock(spec=LLMClient)
        client.settings = MagicMock()
        client.settings.tester_model = "test-model"
        client.settings.workspace = MagicMock()
        client.settings.workspace.exists = MagicMock(return_value=False)
        client.complete = AsyncMock(
            return_value=json.dumps(
                {
                    "passed": True,
                    "summary": "ok",
                    "failures": [],
                }
            )
        )

        tester = TesterAgent(client=client)
        # _detect_test_commands returns empty when no tools are available
        with patch.object(tester, "_detect_test_commands", return_value=[]):
            output = await tester._run_tests()
        assert "No test runner found" in output
