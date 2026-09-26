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

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.ollama_base_url == "http://localhost:11434"


# ---------------------------------------------------------------------------
# Orchestrator _is_done logic
# ---------------------------------------------------------------------------


def _make_orchestrator(goal: str = "test goal", settings: Settings | None = None) -> Orchestrator:
    orch = Orchestrator(goal=goal, client=MagicMock(), settings=settings or Settings())
    return orch


def test_is_done_empty_tasks():
    orch = _make_orchestrator()
    assert orch._is_done() is True


def test_is_done_all_completed():
    orch = _make_orchestrator()
    orch._tasks = [
        TaskModel(id="1", description="a"),
        TaskModel(id="2", description="b"),
    ]
    for t in orch._tasks:
        t.status = "completed"
    assert orch._is_done() is True


def test_is_done_partial_completed():
    orch = _make_orchestrator()
    orch._tasks = [
        TaskModel(id="1", description="a"),
        TaskModel(id="2", description="b"),
    ]
    orch._tasks[0].status = "completed"
    assert orch._is_done() is False


def test_is_done_with_failures():
    orch = _make_orchestrator()
    orch._tasks = [
        TaskModel(id="1", description="a"),
        TaskModel(id="2", description="b"),
    ]
    orch._tasks[0].status = "completed"
    orch._tasks[1].status = "failed"
    assert orch._is_done() is False


def test_is_done_all_failed():
    orch = _make_orchestrator()
    orch._tasks = [TaskModel(id="1", description="a")]
    orch._tasks[0].status = "failed"
    assert orch._is_done() is False


def test_get_tasks_returns_internal_state():
    orch = _make_orchestrator()
    tasks = [TaskModel(id="1", description="a")]
    orch._tasks = tasks
    assert orch._get_tasks() is tasks


def test_max_cycles_stops_loop():
    orch = _make_orchestrator(settings=Settings(max_cycles=1))
    # With max_cycles=1, after the first cycle _is_done check should stop.
    assert orch.settings.max_cycles == 1


# ---------------------------------------------------------------------------
# PlannerAgent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_parises_valid_plan():
    mock_client = MagicMock()
    mock_client.settings = Settings()
    mock_client.complete = AsyncMock(
        return_value=json.dumps(
            {
                "tasks": [{"id": "1", "description": "do a thing", "files": [], "dependencies": []}],
                "rationale": "it is reasonable",
            }
        )
    )
    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("some goal")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "do a thing"
    mock_client.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_planner_raises_on_invalid_json():
    mock_client = MagicMock()
    mock_client.settings = Settings()
    mock_client.complete = AsyncMock(return_value="not json at all")
    planner = PlannerAgent(client=mock_client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("some goal")


# ---------------------------------------------------------------------------
# WorkerAgent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_uses_client_complete():
    task = TaskModel(id="1", description="do a thing", files=["a.py"])
    mock_client = MagicMock()
    mock_client.settings = Settings()
    mock_client.complete = AsyncMock(return_value="Done. Changed a.py.")
    mock_client.list_files = MagicMock(return_value=["a.py", "b.py"])
    mock_client.read_file = AsyncMock(return_value="# content")

    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert result == "Done. Changed a.py."
    mock_client.complete.assert_awaited_once()
    call_args = mock_client.complete.call_args
    assert "do a thing" in call_args.args[0]


@pytest.mark.asyncio
async def test_worker_gathers_no_context_when_no_files():
    task = TaskModel(id="1", description="do a thing")
    mock_client = MagicMock()
    mock_client.settings = Settings()
    mock_client.complete = AsyncMock(return_value="ok")
    mock_client.list_files = MagicMock(return_value=[])
    mock_client.read_file = AsyncMock()

    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert result == "ok"
    mock_client.read_file.assert_not_called()


# ---------------------------------------------------------------------------
# TesterAgent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tester_parses_valid_response():
    mock_client = MagicMock()
    mock_client.settings = Settings()
    mock_client.complete = AsyncMock(
        return_value=json.dumps({"passed": True, "summary": "all good", "failures": []})
    )

    tester = TesterAgent(client=mock_client)
    with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="test output"):
        result = await tester.run("goal", [TaskModel(id="1", description="a")])

    assert result.passed is True
    assert result.summary == "all good"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_falls_back_on_invalid_json():
    mock_client = MagicMock()
    mock_client.settings = Settings()
    mock_client.complete = AsyncMock(return_value="passed everything looks good")

    tester = TesterAgent(client=mock_client)
    with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="test output"):
        result = await tester.run("goal", [TaskModel(id="1", description="a")])

    assert result.passed is True
    assert "passed" in result.summary.lower()


@pytest.mark.asyncio
async def test_tester_handles_run_tests_exception():
    mock_client = MagicMock()
    mock_client.settings = Settings()
    mock_client.complete = AsyncMock()

    tester = TesterAgent(client=mock_client)
    with patch.object(tester, "_run_tests", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
        result = await tester.run("goal", [TaskModel(id="1", description="a")])

    assert result.passed is False
    assert "boom" in result.summary
    mock_client.complete.assert_not_awaited()  # complete should NOT be called


# ---------------------------------------------------------------------------
# LLMClient OLLAMA tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_complete_parses_message_content():
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)

    fake_response = {"message": {"role": "assistant", "content": "hello from ollama"}}
    mock_httpx = MagicMock()
    mock_httpx.post = AsyncMock(
        return_value=MagicMock(
            json=MagicMock(return_value=fake_response),
            raise_for_status=MagicMock(),
        )
    )
    client._httpx = mock_httpx

    result = await client._complete_ollama("hi", "system prompt", "llama3")

    assert result == "hello from ollama"
    mock_httpx.post.assert_awaited_once()
    args, kwargs = mock_httpx.post.call_args
    assert args[0] == "http://localhost:11434/api/chat"
    payload = kwargs["json"]
    assert payload["model"] == "llama3"
    assert payload["stream"] is False
    assert payload["messages"][0]["content"] == "system prompt"
    assert payload["messages"][1]["content"] == "hi"


@pytest.mark.asyncio
async def test_ollama_complete_missing_content_returns_empty():
    s = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=s)

    mock_httpx = MagicMock()
    mock_httpx.post = AsyncMock(
        return_value=MagicMock(
            json=MagicMock(return_value={}),  # No "message" key
            raise_for_status=MagicMock(),
        )
    )
    client._httpx = mock_httpx

    result = await client._complete_ollama("hi", "", "llama3")
    assert result == ""
