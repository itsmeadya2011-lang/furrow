import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


# --- LLMClient tests ---


class TestLLMClient:
    def test_provider_anthropic_routes_to_anthropic(self):
        settings = Settings(provider=Provider.ANTHROPIC)
        client = LLMClient(settings=settings)
        assert client.settings.provider == Provider.ANTHROPIC

    def test_provider_openai_routes_to_openai(self):
        settings = Settings(provider=Provider.OPENAI)
        client = LLMClient(settings=settings)
        assert client.settings.provider == Provider.OPENAI

    def test_provider_ollama_routes_to_ollama(self):
        settings = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=settings)
        assert client.settings.provider == Provider.OLLAMA

    def test_complete_calls_anthropic(self):
        settings = Settings(provider=Provider.ANTHROPIC)
        client = LLMClient(settings=settings)
        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock:
            mock.return_value = "hello"
            result = asyncio.run(client.complete("prompt"))
        assert result == "hello"
        mock.assert_called_once_with("prompt", "", settings.model)

    def test_complete_calls_openai(self):
        settings = Settings(provider=Provider.OPENAI)
        client = LLMClient(settings=settings)
        with patch.object(client, "_complete_openai", new_callable=AsyncMock) as mock:
            mock.return_value = "hello"
            result = asyncio.run(client.complete("prompt"))
        assert result == "hello"
        mock.assert_called_once_with("prompt", "", settings.model)

    def test_complete_calls_ollama(self):
        settings = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=settings)
        with patch.object(client, "_complete_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = "hello"
            result = asyncio.run(client.complete("prompt"))
        assert result == "hello"
        mock.assert_called_once_with("prompt", "", settings.model)

    def test_complete_unsupported_provider(self):
        settings = Settings(provider=Provider.ANTHROPIC)
        settings.provider = "unknown"
        client = LLMClient(settings=settings)
        with pytest.raises(ValueError, match="Unsupported provider"):
            asyncio.run(client.complete("prompt"))

    def test_ollama_url_construction(self):
        settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
        client = LLMClient(settings=settings)
        assert client.settings.ollama_base_url == "http://localhost:11434"

    def test_read_file_and_write_file(self, tmp_path):
        client = LLMClient()
        p = tmp_path / "test.txt"
        asyncio.run(client.write_file(str(p), "hello world"))
        content = asyncio.run(client.read_file(str(p)))
        assert content == "hello world"

    def test_list_files(self, tmp_path):
        client = LLMClient()
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.py").write_text("")
        files = client.list_files(str(tmp_path))
        assert len(files) == 3
        assert "a.py" in files

    def test_retry_on_connection_error(self):
        settings = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=settings)
        with patch("httpx.AsyncClient") as mock_http:
            mock_instance = MagicMock()
            mock_instance.post = AsyncMock(side_effect=ConnectionError("fail"))
            mock_http.return_value = mock_instance
            with pytest.raises(ConnectionError):
                asyncio.run(client._complete_ollama("prompt", "", "model"))


# --- WorkerAgent tests ---


class TestWorkerAgent:
    def test_worker_run_returns_string(self):
        task = TaskModel(id="1", description="test task")
        client = MagicMock()
        client.complete = AsyncMock(return_value="done")
        client.settings.worker_model = "test-model"
        worker = WorkerAgent(task=task, client=client)
        result = asyncio.run(worker.run())
        assert result == "done"

    def test_worker_read_file(self):
        task = TaskModel(id="1", description="test")
        client = MagicMock()
        client.read_file = AsyncMock(return_value="file content")
        client.settings.worker_model = "test-model"
        worker = WorkerAgent(task=task, client=client)
        result = asyncio.run(worker.read_file("/path/to/file"))
        assert result == "file content"

    def test_worker_write_file(self):
        task = TaskModel(id="1", description="test")
        client = MagicMock()
        client.write_file = AsyncMock()
        client.settings.worker_model = "test-model"
        worker = WorkerAgent(task=task, client=client)
        asyncio.run(worker.write_file("/path/to/file", "content"))
        client.write_file.assert_called_once_with("/path/to/file", "content")

    def test_worker_list_files(self):
        task = TaskModel(id="1", description="test")
        client = MagicMock()
        client.list_files.return_value = ["a.py", "b.py"]
        client.settings.worker_model = "test-model"
        worker = WorkerAgent(task=task, client=client)
        result = worker.list_files("/some/dir")
        assert result == ["a.py", "b.py"]

    def test_worker_execute_code(self):
        task = TaskModel(id="1", description="test")
        client = MagicMock()
        client.settings.worker_model = "test-model"
        worker = WorkerAgent(task=task, client=client)
        result = asyncio.run(worker.execute_code("print('hello')"))
        assert "hello" in result


# --- TesterAgent tests ---


class TestTesterAgent:
    def test_run_with_json_response(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value=json.dumps({"passed": True, "summary": "ok", "failures": []}))
        client.settings.tester_model = "test-model"
        agent = TesterAgent(client=client)
        task = TaskModel(id="1", description="test")
        result = asyncio.run(agent.run("goal", [task]))
        assert result.passed is True
        assert result.summary == "ok"

    def test_run_with_non_json_response(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value="All tests passed")
        client.settings.tester_model = "test-model"
        agent = TesterAgent(client=client)
        task = TaskModel(id="1", description="test")
        result = asyncio.run(agent.run("goal", [task]))
        assert result.passed is True

    def test_run_with_failed_json_response(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value=json.dumps({
            "passed": False,
            "summary": "1 failure",
            "failures": ["test_x failed"],
        }))
        client.settings.tester_model = "test-model"
        agent = TesterAgent(client=client)
        task = TaskModel(id="1", description="test")
        result = asyncio.run(agent.run("goal", [task]))
        assert result.passed is False
        assert result.failures == ["test_x failed"]

    def test_run_test_execution_failure(self):
        client = MagicMock()
        agent = TesterAgent(client=client)
        task = TaskModel(id="1", description="test")
        with patch.object(agent, "_run_tests", side_effect=RuntimeError("no tests")):
            result = asyncio.run(agent.run("goal", [task]))
        assert result.passed is False
        assert "no tests" in result.summary

    def test_run_llm_failure(self):
        client = MagicMock()
        client.complete = AsyncMock(side_effect=RuntimeError("LLM error"))
        client.settings.tester_model = "test-model"
        agent = TesterAgent(client=client)
        task = TaskModel(id="1", description="test")
        result = asyncio.run(agent.run("goal", [task]))
        assert result.passed is False
        assert "LLM error" in result.summary


# --- Orchestrator tests ---


class TestOrchestrator:
    def test_is_done_with_all_completed(self):
        client = MagicMock()
        client.settings.worker_model = "test"
        client.settings.tester_model = "test"
        orchestrator = Orchestrator(goal="test goal", client=client)
        orchestrator._plan_tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="completed"),
        ]
        assert orchestrator._is_done() is True

    def test_is_done_with_failed_task(self):
        client = MagicMock()
        client.settings.worker_model = "test"
        client.settings.tester_model = "test"
        orchestrator = Orchestrator(goal="test goal", client=client)
        orchestrator._plan_tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="failed"),
        ]
        assert orchestrator._is_done() is False

    def test_is_done_with_partial_completion(self):
        client = MagicMock()
        client.settings.worker_model = "test"
        client.settings.tester_model = "test"
        orchestrator = Orchestrator(goal="test goal", client=client)
        orchestrator._plan_tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="pending"),
        ]
        assert orchestrator._is_done() is False

    def test_is_done_with_no_tasks(self):
        client = MagicMock()
        client.settings.worker_model = "test"
        client.settings.tester_model = "test"
        orchestrator = Orchestrator(goal="test goal", client=client)
        assert orchestrator._is_done() is True

    def test_get_tasks_returns_persisted_state(self):
        client = MagicMock()
        client.settings.worker_model = "test"
        client.settings.tester_model = "test"
        orchestrator = Orchestrator(goal="test goal", client=client)
        assert orchestrator._get_tasks() == []
        orchestrator._plan_tasks = [TaskModel(id="1", description="t1", status="pending")]
        assert len(orchestrator._get_tasks()) == 1

    def test_max_cycles_limits_cycles(self):
        client = MagicMock()
        client.settings.worker_model = "test"
        client.settings.tester_model = "test"
        with patch("furrow.config.settings.max_cycles", 2):
            with patch("furrow.config.settings.max_parallel_tasks", 5):
                orchestrator = Orchestrator(goal="test goal", client=client)
                with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock) as mock_plan:
                    mock_plan.return_value = Plan(
                        tasks=[TaskModel(id="1", description="t1", status="completed")],
                        rationale="test",
                    )
                    asyncio.run(orchestrator.run())
                    assert orchestrator.cycles == 3

    def test_max_parallel_tasks_is_respected(self):
        with patch("furrow.config.settings.max_parallel_tasks", 2):
            with patch("furrow.config.settings.max_cycles", 0):
                client = MagicMock()
                client.settings.worker_model = "test"
                client.settings.tester_model = "test"
                orchestrator = Orchestrator(goal="test", client=client)
                with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock) as mock_plan:
                    tasks = [
                        TaskModel(id=str(i), description=f"t{i}", status="pending")
                        for i in range(5)
                    ]
                    mock_plan.return_value = Plan(tasks=tasks, rationale="test")
                    with patch.object(WorkerAgent, "run", new_callable=AsyncMock) as mock_run:
                        mock_run.return_value = "done"
                        asyncio.run(orchestrator.run())
                        assert len(mock_run.call_args_list) == 5

    def test_cycle_with_no_tasks_returns(self):
        client = MagicMock()
        client.settings.worker_model = "test"
        client.settings.tester_model = "test"
        with patch("furrow.config.settings.max_cycles", 0):
            with patch("furrow.config.settings.max_parallel_tasks", 5):
                orchestrator = Orchestrator(goal="test", client=client)
                with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock) as mock_plan:
                    mock_plan.return_value = Plan(tasks=[], rationale="nothing")
                    asyncio.run(orchestrator.run())
                    assert orchestrator.cycles == 1

    def test_failed_tests_trigger_goal_update(self):
        client = MagicMock()
        client.settings.worker_model = "test"
        client.settings.tester_model = "test"
        with patch("furrow.config.settings.max_cycles", 0):
            with patch("furrow.config.settings.max_parallel_tasks", 5):
                orchestrator = Orchestrator(goal="test", client=client)
                with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock) as mock_plan:
                    mock_plan.return_value = Plan(
                        tasks=[TaskModel(id="1", description="t1", status="completed")],
                        rationale="test",
                    )
                    with patch.object(TesterAgent, "run", new_callable=AsyncMock) as mock_test:
                        mock_test.return_value = TestResult(
                            passed=False,
                            summary="fail",
                            failures=["test_a failed", "test_b failed"],
                        )
                        asyncio.run(orchestrator.run())
                        assert "test_a failed" in orchestrator.goal
                        assert "test_b failed" in orchestrator.goal


# --- Config tests ---


class TestSettings:
    def test_default_settings(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0
        assert s.log_level == "INFO"

    def test_ollama_provider(self):
        s = Settings(provider=Provider.OLLAMA)
        assert s.provider == Provider.OLLAMA
        assert s.ollama_base_url == "http://localhost:11434"

    def test_model_override(self):
        s = Settings(model="gpt-4")
        assert s.model == "gpt-4"


# --- Plan/Task/TestResult model tests ---


class TestModels:
    def test_task_model_defaults(self):
        t = TaskModel(id="1", description="test")
        assert t.status == "pending"
        assert t.result is None
        assert t.files == []
        assert t.dependencies == []

    def test_plan_model(self):
        tasks = [TaskModel(id="1", description="t1")]
        p = Plan(tasks=tasks, rationale="test")
        assert len(p.tasks) == 1
        assert p.rationale == "test"

    def test_test_result_passed(self):
        t = TestResult(passed=True, summary="ok")
        assert t.passed is True
        assert t.failures == []

    def test_test_result_failed(self):
        t = TestResult(passed=False, summary="fail", failures=["e1"])
        assert t.passed is False
        assert len(t.failures) == 1
