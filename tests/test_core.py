import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from furrow.agents.tester import detect_test_commands
from furrow.config import Plan, Provider, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_default_status():
    task = TaskModel(id="1", description="test task")
    assert task.status == "pending"
    assert task.result is None
    assert task.files == []
    assert task.dependencies == []


def test_plan_with_multiple_tasks():
    tasks = [
        TaskModel(id="1", description="first task"),
        TaskModel(id="2", description="second task"),
        TaskModel(id="3", description="third task"),
    ]
    plan = Plan(tasks=tasks, rationale="test plan")
    assert len(plan.tasks) == 3
    assert plan.rationale == "test plan"


def test_test_result_with_failures():
    result = TestResult(
        passed=False,
        summary="Tests failed",
        failures=["test_a failed", "test_b failed"],
    )
    assert result.passed is False
    assert len(result.failures) == 2


def test_detect_test_commands_python(tmp_path):
    # Create a pyproject.toml
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    commands = detect_test_commands(tmp_path)
    assert commands == [["pytest", "-q"], ["python", "-m", "pytest", "-q"]]


def test_detect_test_commands_node(tmp_path):
    # Create a package.json
    (tmp_path / "package.json").write_text('{"name": "test"}')
    commands = detect_test_commands(tmp_path)
    assert commands == [
        ["npm", "test", "--", "--silent"],
        ["pnpm", "test", "--", "--silent"],
        ["yarn", "test", "--silent"],
    ]


def test_detect_test_commands_rust(tmp_path):
    # Create a Cargo.toml
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'test'\n")
    commands = detect_test_commands(tmp_path)
    assert commands == [["cargo", "test", "-q"]]


def test_detect_test_commands_go(tmp_path):
    # Create a go.mod
    (tmp_path / "go.mod").write_text("module test\n")
    commands = detect_test_commands(tmp_path)
    assert commands == [["go", "test", "./..."]]


def test_detect_test_commands_fallback(tmp_path):
    # No manifest files - should return all commands
    commands = detect_test_commands(tmp_path)
    assert len(commands) > 0
    # Should include pytest as fallback
    assert ["pytest", "-q"] in commands


def test_orchestrator_is_done_no_tasks():
    orchestrator = Orchestrator(goal="test")
    orchestrator._current_tasks = []
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_all_completed():
    orchestrator = Orchestrator(goal="test")
    orchestrator._current_tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_with_failures():
    orchestrator = Orchestrator(goal="test")
    orchestrator._current_tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="failed"),
    ]
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_partial_completion():
    orchestrator = Orchestrator(goal="test")
    orchestrator._current_tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="pending"),
    ]
    assert orchestrator._is_done() is False


def test_orchestrator_max_cycles():
    orchestrator = Orchestrator(goal="test", max_cycles=3)
    assert orchestrator.max_cycles == 3


def test_orchestrator_max_parallel_tasks():
    orchestrator = Orchestrator(goal="test", max_parallel_tasks=2)
    assert orchestrator.max_parallel_tasks == 2


@pytest.mark.asyncio
async def test_orchestrator_emits_output():
    """Test that the orchestrator calls the on_output callback."""
    messages = []

    async def capture_output(msg: str) -> None:
        messages.append(msg)

    orchestrator = Orchestrator(goal="test", on_output=capture_output)
    # Just verify the callback is set
    await orchestrator._emit("test message")
    assert "test message" in messages


def test_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"
