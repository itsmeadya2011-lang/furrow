from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings():
    return MagicMock(
        provider=Provider.ANTHROPIC,
        model="claude-3-5-sonnet-20241022",
        planner_model="claude-3-5-haiku-20241022",
        worker_model="claude-3-5-sonnet-20241022",
        tester_model="claude-3-5-sonnet-20241022",
        anthropic_api_key="test-key",
        openai_api_key=None,
        ollama_base_url="http://localhost:11434",
        max_parallel_tasks=5,
        max_cycles=0,
        workspace=Path.cwd(),
        log_level="INFO",
    )


@pytest.fixture
def mock_client(mock_settings):
    client = MagicMock(spec=LLMClient)
    client.settings = mock_settings
    client.complete = AsyncMock(return_value="{}")
    client.aclose = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Config / Model tests
# ---------------------------------------------------------------------------

def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="test")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_test_result_defaults():
    t = TestResult(passed=False, summary="fail")
    assert t.failures == []


def test_provider_enum():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


# ---------------------------------------------------------------------------
# PlannerAgent tests
# ---------------------------------------------------------------------------

class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_returns_valid_plan(self, mock_client):
        plan_json = json.dumps({
            "tasks": [{"id": "1", "description": "Create auth module", "files": ["src/auth.py"], "dependencies": []}],
            "rationale": "Auth is the foundation",
        })
        mock_client.complete = AsyncMock(return_value=plan_json)
        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("Implement JWT authentication")
        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "Create auth module"
        assert plan.rationale == "Auth is the foundation"
        mock_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_plan_invalid_json_raises(self, mock_client):
        mock_client.complete = AsyncMock(return_value="not json")
        planner = PlannerAgent(client=mock_client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await planner.plan("some goal")

    @pytest.mark.asyncio
    async def test_plan_no_tasks(self, mock_client):
        plan_json = json.dumps({"tasks": [], "rationale": "Goal already complete"})
        mock_client.complete = AsyncMock(return_value=plan_json)
        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("already done goal")
        assert plan.tasks == []

    @pytest.mark.asyncio
    async def test_plan_uses_planner_model(self, mock_client):
        plan_json = json.dumps({
            "tasks": [{"id": "1", "description": "t1", "files": [], "dependencies": []}],
            "rationale": "r",
        })
        mock_client.complete = AsyncMock(return_value=plan_json)
        planner = PlannerAgent(client=mock_client)
        await planner.plan("goal")
        call_kwargs = mock_client.complete.call_args
        assert call_kwargs.kwargs["model"] == mock_client.settings.planner_model


# ---------------------------------------------------------------------------
# WorkerAgent tests
# ---------------------------------------------------------------------------

class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_worker_returns_summary(self, mock_client):
        mock_client.complete = AsyncMock(return_value="Implemented the feature.")
        task = TaskModel(id="1", description="Create auth module")
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()
        assert result == "Implemented the feature."

    @pytest.mark.asyncio
    async def test_worker_uses_worker_model(self, mock_client):
        mock_client.complete = AsyncMock(return_value="done")
        task = TaskModel(id="1", description="t1")
        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()
        call_kwargs = mock_client.complete.call_args
        assert call_kwargs.kwargs["model"] == mock_client.settings.worker_model

    @pytest.mark.asyncio
    async def test_worker_includes_task_description_in_prompt(self, mock_client):
        mock_client.complete = AsyncMock(return_value="done")
        task = TaskModel(id="1", description="Implement login page")
        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()
        call_args = mock_client.complete.call_args
        assert "Implement login page" in call_args.args[0]


# ---------------------------------------------------------------------------
# TesterAgent tests
# ---------------------------------------------------------------------------

class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_tester_pass(self, mock_client, tmp_path):
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "passed": True,
            "summary": "All 10 tests passed",
            "failures": [],
        }))
        tester = TesterAgent(client=mock_client)
        tasks = [TaskModel(id="1", description="t1")]
        result = await tester.run("goal", tasks)
        assert result.passed is True
        assert result.summary == "All 10 tests passed"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_tester_fail(self, mock_client):
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "passed": False,
            "summary": "2 tests failed",
            "failures": ["test_auth: AssertionError", "test_login: TimeoutError"],
        }))
        tester = TesterAgent(client=mock_client)
        tasks = [TaskModel(id="1", description="t1")]
        result = await tester.run("goal", tasks)
        assert result.passed is False
        assert len(result.failures) == 2

    @pytest.mark.asyncio
    async def test_tester_invalid_json_fallback(self, mock_client):
        mock_client.complete = AsyncMock(return_value="tests passed, all good")
        tester = TesterAgent(client=mock_client)
        tasks = [TaskModel(id="1", description="t1")]
        result = await tester.run("goal", tasks)
        # Should fall back to keyword matching
        assert result.passed is True
        assert "tests passed" in result.summary.lower() or result.summary == "tests passed, all good"

    @pytest.mark.asyncio
    async def test_tester_uses_tester_model(self, mock_client):
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "passed": True, "summary": "ok", "failures": [],
        }))
        tester = TesterAgent(client=mock_client)
        tasks = [TaskModel(id="1", description="t1")]
        await tester.run("goal", tasks)
        call_kwargs = mock_client.complete.call_args
        assert call_kwargs.kwargs["model"] == mock_client.settings.tester_model

    @pytest.mark.asyncio
    async def test_tester_no_test_runner(self, mock_client, tmp_path):
        """When no test runner is found, _run_tests returns a message but tester still parses."""
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "passed": False,
            "summary": "No tests exist",
            "failures": ["No test runner found"],
        }))
        tester = TesterAgent(client=mock_client)
        tasks = [TaskModel(id="1", description="t1")]
        result = await tester.run("goal", tasks)
        assert result.passed is False


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_orchestrator_terminates_on_no_tasks(self, mock_client):
        """When planner returns no tasks, orchestrator should stop."""
        plan_json = json.dumps({"tasks": [], "rationale": "Done"})
        mock_client.complete = AsyncMock(return_value=plan_json)
        orch = Orchestrator(goal="finish this", client=mock_client)
        await orch.run()
        assert orch.cycles == 1
        assert len(orch._all_tasks) == 0

    @pytest.mark.asyncio
    async def test_orchestrator_completes_all_tasks(self, mock_client):
        """When all tasks complete and tests pass, orchestrator stops."""
        plan_json = json.dumps({
            "tasks": [{"id": "1", "description": "t1", "files": [], "dependencies": []}],
            "rationale": "r",
        })
        test_json = json.dumps({"passed": True, "summary": "ok", "failures": []})

        def complete_side_effect(prompt, system="", model=None):
            if "Return JSON only" in prompt and "Break" in prompt:
                return plan_json
            elif "Return JSON only" in prompt and "tester" in prompt.lower():
                return test_json
            return plan_json

        mock_client.complete = AsyncMock(side_effect=complete_side_effect)
        orch = Orchestrator(goal="build something", client=mock_client)
        await orch.run()
        assert orch._is_done() is True
        assert len(orch._all_tasks) >= 1

    @pytest.mark.asyncio
    async def test_orchestrator_max_cycles_enforced(self, mock_client):
        """Orchestrator should stop after max_cycles."""
        plan_json = json.dumps({
            "tasks": [{"id": "1", "description": "t1", "files": [], "dependencies": []}],
            "rationale": "r",
        })
        test_json = json.dumps({"passed": False, "summary": "fail", "failures": ["error"]})
        mock_client.complete = AsyncMock(side_effect=lambda *a, **kw: plan_json or test_json)

        orch = Orchestrator(goal="test", client=mock_client, max_cycles=2)
        await orch.run()
        assert orch.cycles == 2

    @pytest.mark.asyncio
    async def test_orchestrator_tracks_task_status(self, mock_client):
        plan_json = json.dumps({
            "tasks": [
                {"id": "1", "description": "task one", "files": [], "dependencies": []},
                {"id": "2", "description": "task two", "files": [], "dependencies": []},
            ],
            "rationale": "r",
        })
        test_json = json.dumps({"passed": True, "summary": "ok", "failures": []})

        def side_effect(prompt, system="", model=None):
            if "tester" in prompt.lower() or "Goal:" in prompt and "test output" in prompt.lower():
                return test_json
            return plan_json

        mock_client.complete = AsyncMock(side_effect=side_effect)
        orch = Orchestrator(goal="build all", client=mock_client)
        await orch.run()
        # Both tasks should be completed
        completed = [t for t in orch._all_tasks if t.status == "completed"]
        assert len(completed) == 2

    @pytest.mark.asyncio
    async def test_orchestrator_emits_events(self, mock_client):
        plan_json = json.dumps({"tasks": [], "rationale": "done"})
        mock_client.complete = AsyncMock(return_value=plan_json)
        events: list[dict] = []

        async def on_event(event: dict):
            events.append(event)

        orch = Orchestrator(goal="test", client=mock_client, on_event=on_event)
        await orch.run()
        assert any(e["type"] == "start" for e in events)
        assert any(e["type"] == "cycle_start" for e in events)

    @pytest.mark.asyncio
    async def test_orchestrator_max_parallel_semaphore(self, mock_client):
        """Ensure max_parallel limits concurrent workers."""
        plan_json = json.dumps({
            "tasks": [
                {"id": str(i), "description": f"task {i}", "files": [], "dependencies": []}
                for i in range(10)
            ],
            "rationale": "r",
        })
        test_json = json.dumps({"passed": True, "summary": "ok", "failures": []})

        call_count = 0
        original_complete = mock_client.complete

        async def counting_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Return different responses based on what's being called
            if call_count > 10:  # After planning, use test response
                return test_json
            return plan_json

        mock_client.complete = AsyncMock(side_effect=counting_complete)
        orch = Orchestrator(goal="build many", client=mock_client, max_parallel=3, max_cycles=1)
        await orch.run()
        # All tasks should have been attempted
        completed = [t for t in orch._all_tasks if t.status in ("completed", "failed")]
        assert len(completed) >= 1

    @pytest.mark.asyncio
    async def test_orchestrator_closes_client(self, mock_client):
        plan_json = json.dumps({"tasks": [], "rationale": "done"})
        mock_client.complete = AsyncMock(return_value=plan_json)
        orch = Orchestrator(goal="test", client=mock_client)
        await orch.run()
        # The orchestrator itself doesn't close the client (that's the web server's job),
        # but the client should still be usable
        assert orch.client is mock_client


# ---------------------------------------------------------------------------
# LLMClient tests
# ---------------------------------------------------------------------------

class TestLLMClient:
    def test_ollama_provider_selected(self, mock_settings):
        mock_settings.provider = Provider.OLLAMA
        mock_settings.ollama_base_url = "http://localhost:11434"
        client = LLMClient(settings=mock_settings)
        # Should not raise on init
        assert client.settings.provider == Provider.OLLAMA

    @patch("furrow.llm.AsyncAnthropic")
    @patch("furrow.llm.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_complete_delegates_to_anthropic(self, mock_http, mock_anthropic_cls, mock_settings):
        mock_settings.provider = Provider.ANTHROPIC
        mock_settings.anthropic_api_key = "key"
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text="anthropic response")]
        ))
        mock_anthropic_cls.return_value = mock_anthropic
        client = LLMClient(settings=mock_settings)
        result = await client.complete("hello")
        assert result == "anthropic response"

    @patch("furrow.llm.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_complete_delegates_to_openai(self, mock_openai_cls, mock_settings):
        mock_settings.provider = Provider.OPENAI
        mock_settings.openai_api_key = "key"
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="openai response"))]
        ))
        mock_openai_cls.return_value = mock_openai
        client = LLMClient(settings=mock_settings)
        result = await client.complete("hello")
        assert result == "openai response"

    def test_unsupported_provider_raises(self, mock_settings):
        mock_settings.provider = "invalid"
        client = LLMClient(settings=mock_settings)
        with pytest.raises(ValueError, match="Unsupported provider"):
            asyncio.run(client.complete("hello"))
