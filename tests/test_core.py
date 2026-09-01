from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


# --- Config tests ---

class TestConfig:
    def test_default_settings(self):
        settings = Settings()
        assert settings.provider == Provider.ANTHROPIC
        assert settings.max_cycles == 0
        assert settings.max_parallel_tasks == 5

    def test_task_model_defaults(self):
        task = TaskModel(id="1", description="do thing")
        assert task.files == []
        assert task.dependencies == []
        assert task.status == "pending"
        assert task.result is None

    def test_plan_construction(self):
        plan = Plan(
            tasks=[TaskModel(id="1", description="do thing")],
            rationale="ok",
        )
        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "do thing"

    def test_test_result_defaults(self):
        result = TestResult(passed=True, summary="ok")
        assert result.failures == []

    def test_workspace_path(self):
        settings = Settings()
        assert isinstance(settings.workspace, Path)


# --- Orchestrator tests ---

class TestOrchestrator:
    def test_is_done_no_plan(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator.plan = None
        assert orchestrator._is_done() is True

    def test_is_done_empty_plan(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator.plan = Plan(tasks=[], rationale="empty")
        assert orchestrator._is_done() is True

    def test_is_done_all_completed(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator.plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="completed"),
            ],
            rationale="ok",
        )
        assert orchestrator._is_done() is True

    def test_is_done_some_completed(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator.plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="pending"),
            ],
            rationale="ok",
        )
        assert orchestrator._is_done() is False

    def test_is_done_has_failed(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator.plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="failed"),
            ],
            rationale="ok",
        )
        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_run_stops_at_max_cycles(self):
        client = MagicMock(spec=LLMClient)
        client.settings = Settings(max_cycles=2)
        
        planner = MagicMock(spec=PlannerAgent)
        planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="done"))
        
        orchestrator = Orchestrator(goal="test", client=client)
        orchestrator.planner = planner
        orchestrator.cycles = 0
        
        with patch.object(Orchestrator, "_cycle", new_callable=AsyncMock) as mock_cycle:
            await orchestrator.run()
            assert mock_cycle.call_count == 2


# --- TesterAgent tests ---

class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_run_tests_timeout_kills_process(self):
        agent = TesterAgent(client=MagicMock())
        
        async def fake_communicate():
            await asyncio.sleep(0.1)
            raise asyncio.TimeoutError()
        
        mock_proc = MagicMock()
        mock_proc.communicate = fake_communicate
        mock_proc.kill = MagicMock()
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
                result = await agent._run_tests()
                assert result == "No test runner found."
                mock_proc.kill.assert_called()

    @pytest.mark.asyncio
    async def test_run_returns_test_result(self):
        agent = TesterAgent(client=MagicMock())
        agent.client.complete = AsyncMock(return_value='{"passed": true, "summary": "ok", "failures": []}')
        
        with patch.object(agent, "_run_tests", new_callable=AsyncMock, return_value="all good"):
            result = await agent.run("goal", [])
            assert result.passed is True
            assert result.summary == "ok"


# --- LLMClient tests ---

class TestLLMClient:
    def test_list_files_empty_dir(self, tmp_path: Path):
        client = LLMClient()
        assert client.list_files(tmp_path) == []

    def test_list_files_nonexistent(self):
        client = LLMClient()
        assert client.list_files("/nonexistent/path/12345") == []

    def test_read_write_file(self, tmp_path: Path):
        client = LLMClient()
        target = tmp_path / "test.txt"
        asyncio.run(client.write_file(target, "hello"))
        result = asyncio.run(client.read_file(target))
        assert result == "hello"


# --- WorkerAgent tests ---

class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_run_calls_llm(self):
        client = MagicMock()
        client.settings = Settings()
        client.complete = AsyncMock(return_value="Implemented feature X")
        
        task = TaskModel(id="1", description="Implement X", files=["src/x.py"])
        agent = WorkerAgent(task=task, client=client)
        
        result = await agent.run()
        assert result == "Implemented feature X"
        client.complete.assert_called_once()
        call_args = client.complete.call_args
        assert "Implement X" in call_args[0][0]
        assert "src/x.py" in call_args[0][0]
