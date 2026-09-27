"""Tests for Furrow core components: config models, orchestrator logic, and agent helpers."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult

# ---------------------------------------------------------------------------
# Config / Model tests
# ---------------------------------------------------------------------------


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_test_result_not_collected_by_pytest():
    """Prevent pytest from treating TestResult as a test class."""
    assert TestResult.__test__ is False


def test_task_model_defaults():
    t = TaskModel(id="1", description="x")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


@pytest.fixture
def orchestrator():
    """Create an Orchestrator without requiring API keys."""
    from furrow.core.orchestrator import Orchestrator
    return Orchestrator(goal="test goal")


def test_get_tasks_empty_initially(orchestrator):
    assert orchestrator._get_tasks() == []


def test_task_waves_no_dependencies(orchestrator):
    tasks = [
        TaskModel(id="1", description="a"),
        TaskModel(id="2", description="b"),
    ]
    waves = orchestrator._task_waves(tasks)
    assert len(waves) == 1
    assert {t.id for t in waves[0]} == {"1", "2"}


def test_task_waves_with_dependencies(orchestrator):
    tasks = [
        TaskModel(id="1", description="base"),
        TaskModel(id="2", description="dep on 1", dependencies=["1"]),
        TaskModel(id="3", description="independent"),
    ]
    waves = orchestrator._task_waves(tasks)
    # Wave 1: tasks with no unmet deps (1 and 3)
    # Wave 2: task 2 (depends on 1, which is resolved)
    assert len(waves) == 2
    assert {t.id for t in waves[0]} == {"1", "3"}
    assert {t.id for t in waves[1]} == {"2"}


def test_task_waves_chain_dependencies(orchestrator):
    tasks = [
        TaskModel(id="A", description="a"),
        TaskModel(id="B", description="b", dependencies=["A"]),
        TaskModel(id="C", description="c", dependencies=["B"]),
        TaskModel(id="D", description="d", dependencies=["A"]),
    ]
    waves = orchestrator._task_waves(tasks)
    # Wave 1: A
    # Wave 2: B, D (both depend on A)
    # Wave 3: C (depends on B)
    assert len(waves) == 3
    assert {t.id for t in waves[0]} == {"A"}
    assert {t.id for t in waves[1]} == {"B", "D"}
    assert {t.id for t in waves[2]} == {"C"}


def test_is_done_no_tasks(orchestrator):
    """When no tasks exist, goal is considered done."""
    assert orchestrator._is_done() is True


def test_is_done_all_completed_no_test_result(orchestrator):
    orchestrator.current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    # No test result yet, all tasks completed → done
    assert orchestrator._is_done() is True


def test_is_done_tests_failed(orchestrator):
    orchestrator.current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
    ]
    orchestrator.last_test_result = TestResult(passed=False, summary="fail", failures=["boom"])
    assert orchestrator._is_done() is False


def test_is_done_task_failed(orchestrator):
    orchestrator.current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orchestrator._is_done() is False


def test_is_done_task_pending(orchestrator):
    orchestrator.current_tasks = [
        TaskModel(id="1", description="a", status="pending"),
    ]
    assert orchestrator._is_done() is False


def test_orchestrator_accepts_max_cycles(orchestrator):
    """Verify max_cycles attribute is accessible via settings."""
    # Settings has max_cycles default 0
    assert orchestrator.client.settings.max_cycles == 0


# ---------------------------------------------------------------------------
# LLMClient tests
# ---------------------------------------------------------------------------


def test_llmclient_ollama_routing():
    """Verify OLLAMA provider routes to _complete_ollama."""
    from furrow.llm import LLMClient

    settings = Settings(provider=Provider.OLLAMA, model="llama3")
    client = LLMClient(settings=settings)

    # Check that complete() routes to _complete_ollama
    with patch.object(LLMClient, "_complete_ollama", new=AsyncMock(return_value="ollama response")):
        result = asyncio.run(client.complete("hello", model="llama3"))
        assert result == "ollama response"


def test_llmclient_unsupported_provider():
    """Verify that an unknown provider raises ValueError."""
    from furrow.llm import LLMClient

    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)

    # Temporarily set an invalid provider
    client.settings.provider = "invalid"
    with pytest.raises(ValueError, match="Unsupported provider"):
        asyncio.run(client.complete("hello"))


# ---------------------------------------------------------------------------
# TesterAgent workspace tests
# ---------------------------------------------------------------------------


def test_tester_agent_runs_in_workspace():
    """Verify that TesterAgent runs tests in the configured workspace."""
    from furrow.agents.tester import TesterAgent
    from furrow.config import Settings
    from furrow.llm import LLMClient

    workspace = Path("/tmp/furrow-test-workspace")
    settings = Settings(workspace=workspace)
    client = LLMClient(settings=settings)
    tester = TesterAgent(client=client)

    # Mock the subprocess to capture the cwd that would be used
    captured_cwd = None

    class FakeProc:
        async def communicate(self):
            return (b"test output", b"")
        def kill(self):
            pass

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        nonlocal captured_cwd
        captured_cwd = kwargs.get("cwd")
        return FakeProc()

    with patch("asyncio.create_subprocess_exec", new=fake_create_subprocess_exec), patch.object(
        client, "complete", new=AsyncMock(return_value='{"passed": true, "summary": "ok", "failures": []}')
    ):
        result = asyncio.run(tester.run("goal", []))

    assert captured_cwd is not None
    assert Path(captured_cwd) == workspace
    assert result.passed is True


# ---------------------------------------------------------------------------
# PlannerAgent / WorkerAgent tests (with mocked LLM)
# ---------------------------------------------------------------------------


def test_planner_agent_parses_plan():
    """Verify PlannerAgent parses LLM JSON response into a Plan."""
    from furrow.agents.planner import PlannerAgent
    from furrow.config import Settings
    from furrow.llm import LLMClient

    settings = Settings()
    client = LLMClient(settings=settings)
    planner = PlannerAgent(client=client)

    llm_response = '{"tasks": [{"id": "1", "description": "add feature X", "files": ["x.py"], "dependencies": []}], "rationale": "needs feature X"}'

    with patch.object(client, "complete", new=AsyncMock(return_value=llm_response)):
        plan = asyncio.run(planner.plan("Build feature X"))

    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "add feature X"
    assert plan.rationale == "needs feature X"


def test_planner_agent_invalid_json_raises():
    """Verify PlannerAgent raises on invalid JSON."""
    from furrow.agents.planner import PlannerAgent
    from furrow.config import Settings
    from furrow.llm import LLMClient

    settings = Settings()
    client = LLMClient(settings=settings)
    planner = PlannerAgent(client=client)

    with patch.object(client, "complete", new=AsyncMock(return_value="not json at all")), pytest.raises(
        ValueError, match="Failed to parse plan"
    ):
        asyncio.run(planner.plan("some goal"))


def test_worker_agent_returns_response():
    """Verify WorkerAgent passes task info to the LLM and returns response."""
    from furrow.agents.worker import WorkerAgent
    from furrow.config import Settings, TaskModel
    from furrow.llm import LLMClient

    settings = Settings()
    client = LLMClient(settings=settings)
    task = TaskModel(id="1", description="implement foo", files=["foo.py"])
    worker = WorkerAgent(task=task, client=client)

    with patch.object(client, "complete", new=AsyncMock(return_value="Implemented foo.py")):
        result = asyncio.run(worker.run())

    assert result == "Implemented foo.py"
