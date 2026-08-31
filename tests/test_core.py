import json
from unittest.mock import AsyncMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class TestConfigModels:
    def test_plan_parse(self):
        p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
        assert p.tasks[0].description == "do thing"

    def test_task_model_defaults(self):
        t = TaskModel(id="1", description="do thing")
        assert t.files == []
        assert t.dependencies == []
        assert t.status == "pending"
        assert t.result is None

    def test_task_model_with_values(self):
        t = TaskModel(
            id="1",
            description="do thing",
            files=["a.py"],
            dependencies=["0"],
            status="completed",
            result="done",
        )
        assert t.files == ["a.py"]
        assert t.dependencies == ["0"]
        assert t.status == "completed"
        assert t.result == "done"

    def test_plan_from_dict(self):
        data = {"tasks": [{"id": "1", "description": "x"}], "rationale": "ok"}
        p = Plan(**data)
        assert p.rationale == "ok"
        assert len(p.tasks) == 1

    def test_test_result_defaults(self):
        t = TestResult(passed=True, summary="ok")
        assert t.failures == []

    def test_test_result_with_values(self):
        t = TestResult(passed=False, summary="bad", failures=["err"])
        assert t.passed is False
        assert t.failures == ["err"]

    def test_test_result_from_dict(self):
        t = TestResult(passed=True, summary="all good", failures=[])
        assert t.passed is True


class TestSettings:
    def test_settings_defaults(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.model == "claude-sonnet-4-20250514"
        assert s.planner_model == "claude-3-5-haiku-20241022"
        assert s.worker_model == "claude-3-5-sonnet-20241022"
        assert s.tester_model == "claude-3-5-sonnet-20241022"
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0
        assert s.ollama_base_url == "http://localhost:11434"
        assert s.log_level == "INFO"

    def test_settings_provider_openai(self):
        s = Settings(provider=Provider.OPENAI)
        assert s.provider == Provider.OPENAI

    def test_settings_provider_ollama(self):
        s = Settings(provider=Provider.OLLAMA)
        assert s.provider == Provider.OLLAMA

    def test_settings_custom_values(self):
        s = Settings(provider=Provider.OPENAI, model="gpt-4", max_parallel_tasks=10)
        assert s.provider == Provider.OPENAI
        assert s.model == "gpt-4"
        assert s.max_parallel_tasks == 10


class TestOrchestratorIsDone:
    def test_no_tasks(self):
        o = Orchestrator("goal")
        o.plan = Plan(tasks=[], rationale="none")
        assert o._is_done() is False

    def test_all_completed(self):
        o = Orchestrator("goal")
        o.plan = Plan(
            tasks=[TaskModel(id="1", description="x", status="completed")],
            rationale="ok",
        )
        assert o._is_done() is True

    def test_any_failed(self):
        o = Orchestrator("goal")
        o.plan = Plan(
            tasks=[
                TaskModel(id="1", description="x", status="completed"),
                TaskModel(id="2", description="y", status="failed"),
            ],
            rationale="ok",
        )
        assert o._is_done() is False

    def test_mixed_completed_pending(self):
        o = Orchestrator("goal")
        o.plan = Plan(
            tasks=[
                TaskModel(id="1", description="x", status="completed"),
                TaskModel(id="2", description="y", status="pending"),
            ],
            rationale="ok",
        )
        assert o._is_done() is False


class TestOrchestratorGetTasks:
    def test_plan_set(self):
        o = Orchestrator("goal")
        tasks = [TaskModel(id="1", description="x")]
        o.plan = Plan(tasks=tasks, rationale="ok")
        assert o._get_tasks() == tasks

    def test_plan_none(self):
        o = Orchestrator("goal")
        assert o._get_tasks() == []


class TestLLMClient:
    def test_init_defaults(self):
        client = LLMClient()
        assert client.settings is not None
        assert client._anthropic is None
        assert client._openai is None
        assert client._ollama is None

    def test_init_with_settings(self):
        s = Settings(provider=Provider.OPENAI)
        client = LLMClient(settings=s)
        assert client.settings.provider == Provider.OPENAI


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_run_valid_json(self):
        client = LLMClient()
        client.complete = AsyncMock(
            return_value=json.dumps(
                {
                    "changes": [
                        {"path": "foo.py", "content": "print('hello')"},
                        {"path": "bar.py", "content": "# empty"},
                    ],
                    "summary": "Created files",
                }
            )
        )
        client.write_file = AsyncMock()

        task = TaskModel(id="1", description="create files", files=["foo.py", "bar.py"])
        agent = WorkerAgent(task=task, client=client)
        result = await agent.run()

        assert client.write_file.call_count == 2
        client.write_file.assert_any_call("foo.py", "print('hello')")
        client.write_file.assert_any_call("bar.py", "# empty")
        assert "Created files" in result
        assert "foo.py" in result
        assert "bar.py" in result

    @pytest.mark.asyncio
    async def test_run_invalid_json(self):
        client = LLMClient()
        client.complete = AsyncMock(return_value="this is not json")
        client.write_file = AsyncMock()

        task = TaskModel(id="1", description="do stuff")
        agent = WorkerAgent(task=task, client=client)
        result = await agent.run()

        assert client.write_file.call_count == 0
        assert "Failed to parse LLM response as JSON" in result


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_run_valid_json(self):
        client = LLMClient()
        client.complete = AsyncMock(
            return_value=json.dumps(
                {
                    "passed": True,
                    "summary": "All tests passed",
                    "failures": [],
                }
            )
        )

        agent = TesterAgent(client=client)
        with patch.object(
            agent, "_run_tests", new_callable=AsyncMock, return_value="pytest output"
        ):
            result = await agent.run("goal", [])

        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_run_invalid_json_fallback_passed(self):
        client = LLMClient()
        client.complete = AsyncMock(return_value="tests passed according to output")

        agent = TesterAgent(client=client)
        with patch.object(
            agent, "_run_tests", new_callable=AsyncMock, return_value="output"
        ):
            result = await agent.run("goal", [])

        assert result.passed is True
        assert result.summary == "tests passed according to output"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_run_invalid_json_fallback_failed(self):
        client = LLMClient()
        client.complete = AsyncMock(return_value="tests failed badly")

        agent = TesterAgent(client=client)
        with patch.object(
            agent, "_run_tests", new_callable=AsyncMock, return_value="output"
        ):
            result = await agent.run("goal", [])

        assert result.passed is False
        assert result.summary == "tests failed badly"
        assert result.failures == []
