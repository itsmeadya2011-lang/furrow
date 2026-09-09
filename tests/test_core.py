from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class TestSettings:
    def test_default_provider(self):
        settings = Settings()
        assert settings.provider == Provider.ANTHROPIC

    def test_anthropic_api_key_validation(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            settings.validate_api_keys()

    def test_openai_api_key_validation(self):
        settings = Settings(provider=Provider.OPENAI, openai_api_key=None)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            settings.validate_api_keys()

    def test_ollama_no_validation_required(self):
        settings = Settings(provider=Provider.OLLAMA)
        settings.validate_api_keys()  # should not raise


class TestLLMClient:
    @pytest.fixture
    def mock_settings(self):
        return Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
            model="test-model",
        )

    @pytest.mark.asyncio
    async def test_complete_anthropic(self, mock_settings):
        client = LLMClient(settings=mock_settings)
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="hello")]
        with patch.object(client.anthropic.messages, "create", new_callable=AsyncMock, return_value=mock_response):
            result = await client.complete("test")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_complete_openai(self, mock_settings):
        mock_settings.provider = Provider.OPENAI
        mock_settings.openai_api_key = "test-openai-key"
        client = LLMClient(settings=mock_settings)
        mock_choice = MagicMock()
        mock_choice.message.content = "openai hello"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        with patch.object(client.openai.chat.completions, "create", new_callable=AsyncMock, return_value=mock_response):
            result = await client.complete("test")
        assert result == "openai hello"

    @pytest.mark.asyncio
    async def test_complete_ollama(self, mock_settings):
        mock_settings.provider = Provider.OLLAMA
        mock_settings.openai_api_key = "ollama"
        client = LLMClient(settings=mock_settings)
        mock_choice = MagicMock()
        mock_choice.message.content = "ollama hello"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        with patch.object(client.openai.chat.completions, "create", new_callable=AsyncMock, return_value=mock_response):
            result = await client.complete("test")
        assert result == "ollama hello"


class TestOrchestrator:
    @pytest.fixture
    def mock_client(self):
        settings = Settings(
            anthropic_api_key="test-key",
            max_cycles=0,
            max_parallel_tasks=2,
        )
        client = MagicMock(spec=LLMClient)
        client.settings = settings
        return client

    @pytest.fixture
    def mock_planner(self):
        planner = MagicMock(spec=PlannerAgent)
        planner.plan = AsyncMock()
        return planner

    @pytest.fixture
    def mock_tester(self):
        tester = MagicMock(spec=TesterAgent)
        tester.run = AsyncMock()
        return tester

    @pytest.mark.asyncio
    async def test_orchestrator_stops_when_tests_pass(self, mock_client, mock_planner, mock_tester):
        plan = Plan(
            tasks=[TaskModel(id="1", description="do something", files=["a.py"])],
            rationale="test plan",
        )
        mock_planner.plan.return_value = plan
        mock_tester.run.return_value = TestResult(passed=True, summary="OK", failures=[])
        mock_client.complete = AsyncMock(return_value="done")

        events: list[str] = []
        orchestrator = Orchestrator(goal="test goal", client=mock_client, on_event=events.append)
        orchestrator.planner = mock_planner
        orchestrator._tasks = plan.tasks

        await orchestrator.run()

        assert orchestrator.cycles == 1
        assert orchestrator._is_done() is True
        assert any("Tests passed" in e for e in events)

    @pytest.mark.asyncio
    async def test_orchestrator_retries_on_test_failure(self, mock_client, mock_planner, mock_tester):
        plan_with_tasks = Plan(
            tasks=[TaskModel(id="1", description="do something", files=["a.py"])],
            rationale="test plan",
        )
        plan_no_tasks = Plan(tasks=[], rationale="all done")
        mock_planner.plan.side_effect = [plan_with_tasks, plan_no_tasks]
        mock_tester.run.return_value = TestResult(passed=False, summary="failed", failures=["error"])
        mock_client.complete = AsyncMock(return_value="done")

        events: list[str] = []
        orchestrator = Orchestrator(goal="test goal", client=mock_client, on_event=events.append)
        orchestrator.planner = mock_planner
        orchestrator._tasks = plan_with_tasks.tasks

        await orchestrator.run()

        assert orchestrator.cycles == 2
        assert orchestrator.goal == "Fix failing tests:\nerror"
        assert orchestrator._is_done() is False
        assert any("Tests failed" in e for e in events)

    @pytest.mark.asyncio
    async def test_orchestrator_respects_max_cycles(self, mock_client, mock_planner, mock_tester):
        settings = Settings(
            anthropic_api_key="test-key",
            max_cycles=2,
            max_parallel_tasks=2,
        )
        mock_client.settings = settings
        plan = Plan(
            tasks=[TaskModel(id="1", description="do something", files=["a.py"])],
            rationale="test plan",
        )
        mock_planner.plan.return_value = plan
        mock_tester.run.return_value = TestResult(passed=True, summary="OK", failures=[])

        orchestrator = Orchestrator(goal="test goal", client=mock_client)
        orchestrator.planner = mock_planner
        orchestrator._tasks = plan.tasks

        await orchestrator.run()

        assert orchestrator.cycles == 2


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_worker_writes_files(self):
        task = TaskModel(id="1", description="create a file", files=["test_output.txt"])
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings.worker_model = "test-model"
        mock_client.complete = AsyncMock(
            return_value=json.dumps({
                "files": {"test_output.txt": "hello world"},
                "summary": "Created file",
            })
        )
        mock_client.write_file = AsyncMock()

        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        mock_client.write_file.assert_called_once_with("test_output.txt", "hello world")
        assert result == "Created file"

    @pytest.mark.asyncio
    async def test_worker_returns_raw_text_on_parse_failure(self):
        task = TaskModel(id="1", description="do something", files=["a.py"])
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings.worker_model = "test-model"
        mock_client.complete = AsyncMock(return_value="not json")
        mock_client.write_file = AsyncMock()

        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        mock_client.write_file.assert_not_called()
        assert result == "not json"


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_tester_parses_json_result(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings.tester_model = "test-model"
        mock_client.complete = AsyncMock(
            return_value=json.dumps({"passed": True, "summary": "OK", "failures": []})
        )

        tester = TesterAgent(client=mock_client)
        result = await tester.run("goal", [])

        assert result.passed is True
        assert result.summary == "OK"

    @pytest.mark.asyncio
    async def test_tester_handles_non_json_response(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings.tester_model = "test-model"
        mock_client.complete = AsyncMock(return_value="tests passed")

        tester = TesterAgent(client=mock_client)
        result = await tester.run("goal", [])

        assert result.passed is True
        assert result.summary == "tests passed"


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_planner_parses_plan(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings.planner_model = "test-model"
        plan_json = json.dumps({
            "tasks": [{"id": "1", "description": "task", "files": [], "dependencies": []}],
            "rationale": "reason",
        })
        mock_client.complete = AsyncMock(return_value=plan_json)

        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("test goal")

        assert len(plan.tasks) == 1
        assert plan.rationale == "reason"

    @pytest.mark.asyncio
    async def test_planner_raises_on_invalid_json(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings.planner_model = "test-model"
        mock_client.complete = AsyncMock(return_value="not json")

        planner = PlannerAgent(client=mock_client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await planner.plan("test goal")
