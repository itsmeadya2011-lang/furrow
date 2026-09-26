from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult, setup_logging
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def make_mock_client(settings: Settings | None = None) -> MagicMock:
    """Create a mock LLMClient with settings and async methods."""
    client = MagicMock()
    client.settings = settings or Settings()
    client.complete = AsyncMock()
    client.complete_with_tools = AsyncMock()
    return client


class TestSettings:
    def test_settings_defaults(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0
        assert s.llm_timeout == 60

    def test_settings_provider_enum(self):
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.OPENAI.value == "openai"
        assert Provider.OLLAMA.value == "ollama"

    def test_setup_logging_does_not_crash(self):
        setup_logging("DEBUG")

    def test_settings_with_provider(self):
        s = Settings(provider=Provider.OPENAI, model="gpt-4")
        assert s.provider == Provider.OPENAI
        assert s.model == "gpt-4"


class TestOrchestratorInit:
    def test_init_attributes(self):
        client = make_mock_client()
        console = Console(file=io.StringIO())
        orch = Orchestrator(goal="fix the bug", client=client, console=console)
        assert orch.original_goal == "fix the bug"
        assert orch.goal == "fix the bug"
        assert orch.client is client
        assert orch.console is console
        assert orch.tasks == []
        assert orch.last_test_result is None
        assert orch.cycles == 0

    def test_init_creates_semaphore(self):
        client = make_mock_client()
        orch = Orchestrator(goal="test", client=client)
        assert orch._semaphore is not None

    def test_init_without_client_uses_default(self):
        orch = Orchestrator(goal="test")
        assert orch.original_goal == "test"
        assert isinstance(orch.client, LLMClient)

    def test_init_without_console(self):
        client = make_mock_client()
        orch = Orchestrator(goal="test", client=client)
        assert isinstance(orch.console, Console)


class TestOrchestratorIsDone:
    def test_is_done_no_tasks(self):
        client = make_mock_client()
        orch = Orchestrator(goal="test", client=client)
        assert orch._is_done() is True

    def test_is_done_all_completed_tests_passed(self):
        client = make_mock_client()
        orch = Orchestrator(goal="test", client=client)
        orch.tasks = []
        orch.last_test_result = TestResult(
            passed=True, summary="All good", failures=[]
        )
        assert orch._is_done() is True

    def test_is_done_with_completed_tasks(self):
        client = make_mock_client()
        orch = Orchestrator(goal="test", client=client)
        task = TaskModel(id="1", description="do something")
        task.status = "completed"
        orch.tasks = [task]
        orch.last_test_result = TestResult(
            passed=True, summary="All good", failures=[]
        )
        assert orch._is_done() is True

    def test_is_done_failed_task(self):
        client = make_mock_client()
        orch = Orchestrator(goal="test", client=client)
        task = TaskModel(id="1", description="do something")
        task.status = "failed"
        orch.tasks = [task]
        assert orch._is_done() is False

    def test_is_done_tests_failed(self):
        client = make_mock_client()
        orch = Orchestrator(goal="test", client=client)
        task = TaskModel(id="1", description="do something")
        task.status = "completed"
        orch.tasks = [task]
        orch.last_test_result = TestResult(
            passed=False, summary="Failed", failures=["error"]
        )
        assert orch._is_done() is False

    def test_is_done_no_test_result(self):
        client = make_mock_client()
        orch = Orchestrator(goal="test", client=client)
        task = TaskModel(id="1", description="do something")
        task.status = "completed"
        orch.tasks = [task]
        orch.last_test_result = None
        assert orch._is_done() is False


class TestOrchestratorMaxCycles:
    def test_max_cycles_zero_infinite(self):
        client = make_mock_client()
        orch = Orchestrator(goal="test", client=client)
        orch.cycles = 100
        assert orch._max_cycles_reached() is False

    def test_max_cycles_with_nonzero(self):
        settings = Settings(max_cycles=3)
        client = make_mock_client(settings=settings)
        orch = Orchestrator(goal="test", client=client)
        orch.cycles = 3
        assert orch._max_cycles_reached() is True

    def test_max_cycles_not_reached(self):
        settings = Settings(max_cycles=5)
        client = make_mock_client(settings=settings)
        orch = Orchestrator(goal="test", client=client)
        orch.cycles = 3
        assert orch._max_cycles_reached() is False


class TestPlannerAgent:
    async def test_plan_valid_json(self):
        client = make_mock_client()
        plan_data = {
            "tasks": [
                {
                    "id": "1",
                    "description": "Implement auth",
                    "files": ["src/auth.py"],
                    "dependencies": [],
                }
            ],
            "rationale": "Add authentication",
        }
        client.complete.return_value = json.dumps(plan_data)
        planner = PlannerAgent(client=client)
        plan = await planner.plan("Add authentication")
        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "Implement auth"
        assert plan.rationale == "Add authentication"

    async def test_plan_invalid_json_raises(self):
        client = make_mock_client()
        client.complete.return_value = "not valid json {{{"
        planner = PlannerAgent(client=client)
        with pytest.raises(ValueError, match="Failed to parse plan JSON"):
            await planner.plan("Do something")

    async def test_plan_uses_planner_model(self):
        client = make_mock_client()
        client.complete.return_value = json.dumps(
            {"tasks": [], "rationale": "nothing to do"}
        )
        planner = PlannerAgent(client=client)
        await planner.plan("test")
        assert client.complete.call_args.kwargs["model"] == client.settings.planner_model


class TestWorkerAgent:
    async def test_worker_run_returns_string(self):
        client = make_mock_client()
        client.complete_with_tools.return_value = "Task completed: fixed the file"
        task = TaskModel(id="1", description="Fix the bug")
        worker = WorkerAgent(task=task, client=client)
        result = await worker.run()
        assert isinstance(result, str)
        assert "Task completed" in result

    async def test_worker_uses_complete_with_tools(self):
        client = make_mock_client()
        client.complete_with_tools.return_value = "done"
        task = TaskModel(id="1", description="Fix bug", files=["src/main.py"])
        worker = WorkerAgent(task=task, client=client)
        await worker.run()
        assert client.complete_with_tools.await_count == 1
        assert client.complete.await_count == 0

    async def test_worker_passes_system_prompt(self):
        client = make_mock_client()
        client.complete_with_tools.return_value = "done"
        task = TaskModel(id="1", description="Fix bug")
        worker = WorkerAgent(task=task, client=client)
        await worker.run()
        call_kwargs = client.complete_with_tools.call_args.kwargs
        assert "system" in call_kwargs
        assert call_kwargs["model"] == client.settings.worker_model


class TestTesterAgent:
    async def test_tester_tests_pass(self):
        client = make_mock_client()
        client.complete.return_value = json.dumps(
            {"passed": True, "summary": "All tests passed", "failures": []}
        )
        tester = TesterAgent(client=client)
        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "1 test passed"
            result = await tester.run("Build feature", [])
        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    async def test_tester_tests_fail(self):
        client = make_mock_client()
        client.complete.return_value = json.dumps(
            {
                "passed": False,
                "summary": "2 tests failed",
                "failures": ["test_one failed", "test_two failed"],
            }
        )
        tester = TesterAgent(client=client)
        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "2 tests failed"
            result = await tester.run("Build feature", [])
        assert result.passed is False
        assert result.summary == "2 tests failed"
        assert len(result.failures) == 2

    async def test_tester_no_runner_found(self):
        client = make_mock_client()
        tester = TesterAgent(client=client)
        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "No test runner found."
            client.complete.return_value = json.dumps(
                {"passed": False, "summary": "No runner", "failures": ["none"]}
            )
            result = await tester.run("Build feature", [])
        assert result.passed is False


class TestLLMClient:
    def test_settings_none_does_not_set_none(self):
        client = LLMClient(settings=None)
        assert client.settings is not None
        assert isinstance(client.settings, Settings)

    def test_get_tools_returns_list(self):
        client = LLMClient()
        tools = client.get_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert all("name" in t or "function" in t for t in tools)

    async def test_read_file(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("hello world")
        client = LLMClient()
        content = await client.read_file(p)
        assert content == "hello world"

    async def test_write_file(self, tmp_path):
        p = tmp_path / "output.txt"
        client = LLMClient()
        await client.write_file(p, "file content")
        assert p.read_text() == "file content"

    async def test_list_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        client = LLMClient()
        files = client.list_files(tmp_path)
        assert "a.txt" in files
        assert "b.txt" in files

    def test_settings_ollama_url(self):
        s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:8080")
        assert s.ollama_base_url == "http://localhost:8080"


class TestOrchestratorFullCycle:
    async def test_full_cycle_mocked(self):
        client = make_mock_client(
            Settings(max_cycles=2, max_parallel_tasks=2)
        )
        plan_data = {
            "tasks": [
                {
                    "id": "1",
                    "description": "Task one",
                    "files": [],
                    "dependencies": [],
                }
            ],
            "rationale": "Plan for task one",
        }
        test_data = {
            "passed": True,
            "summary": "All passed",
            "failures": [],
        }

        with patch.object(PlannerAgent, "plan", new_callable=AsyncMock) as mock_plan, \
             patch.object(WorkerAgent, "run", new_callable=AsyncMock) as mock_worker, \
             patch.object(TesterAgent, "run", new_callable=AsyncMock) as mock_tester:
            mock_plan.return_value = Plan(**plan_data)
            mock_worker.return_value = "Completed task one"
            mock_tester.return_value = TestResult(**test_data)

            console = Console(file=io.StringIO())
            orch = Orchestrator(
                goal="Build a feature", client=client, console=console
            )
            await orch.run()

        assert orch.cycles == 1
        assert len(orch.tasks) == 1
        assert orch.tasks[0].status == "completed"
        assert orch.last_test_result is not None
        assert orch.last_test_result.passed is True
        assert orch.original_goal == "Build a feature"
        mock_plan.assert_awaited_once()
        mock_worker.assert_awaited_once()
        mock_tester.assert_awaited_once()

    async def test_full_cycle_max_cycles_halts(self):
        client = make_mock_client(
            Settings(max_cycles=2, max_parallel_tasks=2)
        )
        console = Console(file=io.StringIO())
        orch = Orchestrator(goal="test", client=client, console=console)

        with patch.object(PlannerAgent, "plan", new_callable=AsyncMock) as mock_plan, \
             patch.object(WorkerAgent, "run", new_callable=AsyncMock) as mock_worker, \
             patch.object(TesterAgent, "run", new_callable=AsyncMock) as mock_tester:
            task = TaskModel(id="1", description="do something")
            plan = Plan(tasks=[task], rationale="plan")
            mock_plan.return_value = plan
            mock_worker.return_value = "completed"
            mock_tester.return_value = TestResult(
                passed=False, summary="FAILED", failures=["broken"]
            )
            await orch.run()

        assert orch._max_cycles_reached() is True
        assert orch.cycles == 2


class TestModels:
    def test_task_model_defaults(self):
        t = TaskModel(id="1", description="do thing")
        assert t.files == []
        assert t.dependencies == []
        assert t.status == "pending"
        assert t.result is None

    def test_plan_model(self):
        p = Plan(
            tasks=[TaskModel(id="1", description="do thing")],
            rationale="ok",
        )
        assert p.tasks[0].description == "do thing"
        assert p.rationale == "ok"

    def test_test_result_model(self):
        t = TestResult(passed=True, summary="ok", failures=[])
        assert t.passed is True
        assert isinstance(t.failures, list)
