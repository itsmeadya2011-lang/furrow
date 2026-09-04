from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_defaults():
    s = Settings()
    assert s.max_cycles == 0
    assert s.test_timeout == 120
    assert s.max_parallel_tasks == 5


class TestOrchestrator:
    def test_get_tasks_returns_current_plan_tasks(self):
        client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test", client=client)
        assert orchestrator._get_tasks() == []

        plan = Plan(tasks=[TaskModel(id="1", description="task")], rationale="ok")
        orchestrator._current_plan = plan
        assert orchestrator._get_tasks() == plan.tasks
        assert len(orchestrator._get_tasks()) == 1

    def test_is_done_no_plan_returns_false(self):
        client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test", client=client)
        assert orchestrator._is_done() is False

    def test_is_done_all_completed_returns_true(self):
        client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test", client=client)
        plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="completed"),
            ],
            rationale="ok",
        )
        orchestrator._current_plan = plan
        assert orchestrator._is_done() is True

    def test_is_done_has_failed_returns_false(self):
        client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test", client=client)
        plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="failed"),
            ],
            rationale="ok",
        )
        orchestrator._current_plan = plan
        assert orchestrator._is_done() is False

    def test_is_done_mixed_status_returns_false(self):
        client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test", client=client)
        plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="pending"),
            ],
            rationale="ok",
        )
        orchestrator._current_plan = plan
        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_run_stops_at_max_cycles(self):
        client = MagicMock(spec=LLMClient)
        client.settings = Settings(max_cycles=2)
        orchestrator = Orchestrator(goal="test", client=client)

        call_count = 0

        async def fake_cycle():
            nonlocal call_count
            call_count += 1
            orchestrator._current_plan = Plan(tasks=[], rationale="done")

        with patch.object(orchestrator, "_cycle", side_effect=fake_cycle):
            await orchestrator.run()

        assert call_count == 2


class TestLLMClient:
    def test_list_files_is_async(self):
        client = LLMClient()
        assert asyncio.iscoroutinefunction(client.list_files) is True

    def test_complete_wraps_exception_with_model_name(self):
        client = LLMClient()
        client.settings.model = "test-model"

        async def fake_complete(*args, **kwargs):
            raise ValueError("something went wrong")

        with patch.object(client, "_complete_anthropic", side_effect=fake_complete):
            with pytest.raises(ValueError, match=r"\[test-model\] something went wrong"):
                asyncio.run(client.complete("prompt", provider="anthropic"))


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_run_tests_nonzero_exit_code(self):
        client = MagicMock(spec=LLMClient)
        client.complete = AsyncMock(return_value='{"passed": true, "summary": "ok", "failures": []}')
        agent = TesterAgent(client=client)

        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(b"output", b""))
        proc.returncode = 1

        mock_process = MagicMock()
        mock_process.return_value = proc

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
                # Timeout path
                pass

            proc.communicate = AsyncMock(return_value=(b"output", b""))
            proc.returncode = 1
            result = await agent._run_tests()
            assert "[Process exited with code 1]" in result
            assert "output" in result

    @pytest.mark.asyncio
    async def test_run_tests_no_runner_found(self):
        client = MagicMock(spec=LLMClient)
        agent = TesterAgent(client=client)

        with patch("asyncio.create_subprocess_exec", side_effect=OSError("not found")):
            result = await agent._run_tests()
            assert result == "No test runner found."
