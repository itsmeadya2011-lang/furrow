import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


class TestOrchestratorDoneLogic:
    def test_get_tasks_returns_empty_when_no_plan(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = Settings()
        orchestrator = Orchestrator("test goal", client=mock_client)
        assert orchestrator._get_tasks() == []

    def test_get_tasks_returns_plan_tasks(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = Settings()
        orchestrator = Orchestrator("test goal", client=mock_client)
        orchestrator.plan = Plan(
            tasks=[TaskModel(id="1", description="test")], rationale="ok"
        )
        assert len(orchestrator._get_tasks()) == 1
        assert orchestrator._get_tasks()[0].id == "1"

    def test_is_done_returns_true_when_no_plan(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = Settings()
        orchestrator = Orchestrator("test goal", client=mock_client)
        assert orchestrator._is_done() is True

    def test_is_done_returns_true_when_all_completed(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = Settings()
        orchestrator = Orchestrator("test goal", client=mock_client)
        orchestrator.plan = Plan(
            tasks=[TaskModel(id="1", description="test", status="completed")],
            rationale="ok",
        )
        assert orchestrator._is_done() is True

    def test_is_done_returns_false_when_any_failed(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = Settings()
        orchestrator = Orchestrator("test goal", client=mock_client)
        orchestrator.plan = Plan(
            tasks=[
                TaskModel(id="1", description="test", status="completed"),
                TaskModel(id="2", description="test", status="failed"),
            ],
            rationale="ok",
        )
        assert orchestrator._is_done() is False

    def test_is_done_returns_false_when_pending(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = Settings()
        orchestrator = Orchestrator("test goal", client=mock_client)
        orchestrator.plan = Plan(
            tasks=[TaskModel(id="1", description="test", status="pending")],
            rationale="ok",
        )
        assert orchestrator._is_done() is False

    def test_is_done_respects_max_cycles(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = Settings(max_cycles=3)
        orchestrator = Orchestrator("test goal", client=mock_client)
        orchestrator.cycles = 3
        assert orchestrator._is_done() is True

    def test_is_done_not_done_below_max_cycles(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = Settings(max_cycles=5)
        orchestrator = Orchestrator("test goal", client=mock_client)
        orchestrator.cycles = 2
        orchestrator.plan = Plan(
            tasks=[TaskModel(id="1", description="test", status="completed")],
            rationale="ok",
        )
        assert orchestrator._is_done() is True  # all completed

    def test_is_done_zero_max_cycles_is_unlimited(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = Settings(max_cycles=0)
        orchestrator = Orchestrator("test goal", client=mock_client)
        orchestrator.cycles = 999
        orchestrator.plan = Plan(
            tasks=[TaskModel(id="1", description="test", status="completed")],
            rationale="ok",
        )
        assert orchestrator._is_done() is True  # all completed, unlimited cycles

    def test_is_done_max_cycles_zero_does_not_halt_on_incomplete(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = Settings(max_cycles=0)
        orchestrator = Orchestrator("test goal", client=mock_client)
        orchestrator.cycles = 999
        orchestrator.plan = Plan(
            tasks=[TaskModel(id="1", description="test", status="pending")],
            rationale="ok",
        )
        assert orchestrator._is_done() is False  # pending tasks, unlimited cycles


class TestTesterExitCodes:
    @patch("asyncio.create_subprocess_exec")
    async def test_run_tests_reports_nonzero_exit(self, mock_subprocess):
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b"some output", b""))
        mock_proc.returncode = 1
        mock_subprocess.return_value = mock_proc

        from furrow.agents.tester import TesterAgent

        mock_client = MagicMock(spec=LLMClient)
        agent = TesterAgent(client=mock_client)

        result = await agent._run_tests()
        assert "exit code 1" in result
        assert "some output" in result

    @patch("asyncio.create_subprocess_exec")
    async def test_run_tests_reports_zero_exit(self, mock_subprocess):
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b"all good", b""))
        mock_proc.returncode = 0
        mock_subprocess.return_value = mock_proc

        from furrow.agents.tester import TesterAgent

        mock_client = MagicMock(spec=LLMClient)
        agent = TesterAgent(client=mock_client)

        result = await agent._run_tests()
        assert result == "all good"

    @patch("asyncio.create_subprocess_exec")
    async def test_run_tests_falls_through_on_not_found(self, mock_subprocess):
        mock_subprocess.side_effect = FileNotFoundError

        from furrow.agents.tester import TesterAgent

        mock_client = MagicMock(spec=LLMClient)
        agent = TesterAgent(client=mock_client)

        result = await agent._run_tests()
        assert result == "No test runner found."

    @patch("asyncio.create_subprocess_exec")
    async def test_run_tests_falls_through_on_timeout(self, mock_subprocess):
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_proc.kill = MagicMock()
        mock_subprocess.return_value = mock_proc

        from furrow.agents.tester import TesterAgent

        mock_client = MagicMock(spec=LLMClient)
        agent = TesterAgent(client=mock_client)

        result = await agent._run_tests()
        assert result == "No test runner found."
        mock_proc.kill.assert_called_once()
