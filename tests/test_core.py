"""Tests for Furrow core: models, json_utils, and orchestrator logic."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.json_utils import extract_json, parse_json_response
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


# --- Helper ---


def _make_orchestrator() -> Orchestrator:
    """Create an Orchestrator instance without making any LLM calls."""
    return Orchestrator(goal="test goal")


# --- json_utils: extract_json ---


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == '{"a": 1}'


def test_extract_json_markdown_fence():
    text = "```json\n{\"a\": 1}\n```"
    assert extract_json(text) == '{"a": 1}'


def test_extract_json_with_prose():
    text = "Here is the result:\n```json\n{\"tasks\": []}\n```\nDone!"
    assert extract_json(text) == '{"tasks": []}'


def test_extract_json_array_syntax():
    assert extract_json("[1, 2, 3]") == "[1, 2, 3]"


def test_extract_json_no_json_returns_original():
    text = "plain text with no json"
    assert extract_json(text) == text


# --- json_utils: parse_json_response ---


def test_parse_json_response_valid():
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_response_markdown_fence():
    text = "```json\n{\n  \"a\": 1\n}\n```"
    assert parse_json_response(text) == {"a": 1}


def test_parse_json_response_invalid_returns_default():
    assert parse_json_response("not json at all", default="fallback") == "fallback"


def test_parse_json_response_invalid_returns_none():
    assert parse_json_response("not json at all") is None


# --- Config / model tests ---


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


def test_settings_defaults():
    s = Settings()
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"
    assert s.ollama_base_url == "http://localhost:11434"


# --- Orchestrator: critical _is_done / _get_tasks fix ---


def test_orchestrator_get_tasks_returns_current():
    """_get_tasks() must return tracked tasks, not a hardcoded empty list."""
    orch = _make_orchestrator()
    assert orch._get_tasks() == []
    task = TaskModel(id="1", description="do something", status="completed")
    orch.current_tasks = [task]
    assert orch._get_tasks() == [task]


def test_is_done_no_tasks_and_tests_passed():
    orch = _make_orchestrator()
    orch._last_tests_passed = True
    orch._no_tasks_planned = True
    assert orch._is_done() is True


def test_is_done_tests_failed():
    orch = _make_orchestrator()
    orch._last_tests_passed = False
    orch._no_tasks_planned = True
    assert orch._is_done() is False


def test_is_done_tasks_remaining_not_done():
    orch = _make_orchestrator()
    orch._last_tests_passed = True
    orch._no_tasks_planned = False
    orch.current_tasks = [TaskModel(id="1", description="do something")]  # pending
    assert orch._is_done() is False


def test_is_done_all_tasks_completed():
    orch = _make_orchestrator()
    orch._last_tests_passed = True
    orch._no_tasks_planned = False
    orch.current_tasks = [
        TaskModel(id="1", description="do something", status="completed")
    ]
    assert orch._is_done() is True


def test_is_done_failed_task_not_done():
    orch = _make_orchestrator()
    orch._last_tests_passed = True
    orch._no_tasks_planned = False
    orch.current_tasks = [TaskModel(id="1", description="do something", status="failed")]
    assert orch._is_done() is False


# --- Orchestrator _emit tests ---


def test_emit_string_callback():
    """_emit should pass strings to on_output callback."""
    orch = _make_orchestrator()
    received = []
    orch.on_output = lambda msg: received.append(msg)
    orch._emit("hello")
    assert received == ["hello"]


def test_emit_strips_rich_tags():
    """_emit should strip Rich markup tags when on_output is set."""
    orch = _make_orchestrator()
    received = []
    orch.on_output = lambda msg: received.append(msg)
    orch._emit("[red]Error occurred[/red]")
    assert received == ["Error occurred"]


def test_emit_renders_rich_objects():
    """_emit should render Rich Panel objects to plain text for callbacks."""
    from rich.panel import Panel

    orch = _make_orchestrator()
    received = []
    orch.on_output = lambda msg: received.append(msg)
    orch._emit(Panel("Hello", title="Test"))
    assert len(received) == 1
    assert isinstance(received[0], str)
    assert "Hello" in received[0]


# --- Orchestrator integration tests (mocked LLM) ---


@pytest.mark.asyncio
async def test_orchestrator_max_cycles():
    """Orchestrator should stop when max_cycles is exceeded, even if not done."""
    settings = Settings(max_cycles=1)
    orch = Orchestrator(goal="test", settings=settings)

    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale=""))
    orch.tester = MagicMock()
    orch.tester.run = AsyncMock(
        return_value=TestResult(passed=False, summary="fail", failures=["error"])
    )

    outputs = []
    orch.on_output = lambda msg: outputs.append(msg)

    await orch.run()

    # max_cycles=1: cycle 1 runs (1 > 1 = False), tests fail -> _is_done False,
    # cycle 2: 2 > 1 = True -> break
    assert orch.cycles == 2


@pytest.mark.asyncio
async def test_orchestrator_stops_when_no_tasks():
    """Orchestrator should stop after one cycle when no tasks are planned and tests pass."""
    settings = Settings()
    orch = Orchestrator(goal="test", settings=settings)

    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="done"))
    orch.tester = MagicMock()
    orch.tester.run = AsyncMock(
        return_value=TestResult(passed=True, summary="ok", failures=[])
    )

    outputs = []
    orch.on_output = lambda msg: outputs.append(msg)

    await orch.run()

    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_orchestrator_continues_when_tests_fail(monkeypatch):
    """Orchestrator should continue after test failures and stop when tests pass."""
    settings = Settings(max_cycles=10)
    orch = Orchestrator(goal="test", settings=settings)

    plan_calls = 0

    async def mock_plan(goal):
        nonlocal plan_calls
        plan_calls += 1
        if plan_calls <= 2:
            return Plan(
                tasks=[TaskModel(id=str(plan_calls), description="fix")], rationale=""
            )
        return Plan(tasks=[], rationale="done")

    test_calls = 0

    async def mock_test(goal, tasks):
        nonlocal test_calls
        test_calls += 1
        if test_calls == 1:
            return TestResult(passed=False, summary="fail", failures=["something broke"])
        return TestResult(passed=True, summary="ok", failures=[])

    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(side_effect=mock_plan)
    orch.tester = MagicMock()
    orch.tester.run = AsyncMock(side_effect=mock_test)

    # Mock WorkerAgent to avoid LLM calls
    mock_worker_inst = MagicMock()
    mock_worker_inst.run = AsyncMock(return_value="Task completed successfully")
    mock_worker_cls = MagicMock(return_value=mock_worker_inst)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", mock_worker_cls)

    outputs = []
    orch.on_output = lambda msg: outputs.append(msg)

    await orch.run()

    # Cycle 1: tests fail -> continue. Cycle 2: tests pass + task completed -> done.
    assert orch.cycles == 2
    assert orch._last_tests_passed is True
