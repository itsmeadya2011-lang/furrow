import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_with_multiple_tasks():
    p = Plan(
        tasks=[
            TaskModel(id="1", description="first", files=["a.py"]),
            TaskModel(id="2", description="second", dependencies=["1"]),
        ],
        rationale="do things",
    )
    assert len(p.tasks) == 2
    assert p.tasks[0].files == ["a.py"]
    assert p.tasks[1].dependencies == ["1"]


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="bad", failures=["error1", "error2"])
    assert t.passed is False
    assert len(t.failures) == 2


def test_orchestrator_get_tasks():
    o = Orchestrator.__new__(Orchestrator)
    tasks = [TaskModel(id="1", description="task")]
    o._current_tasks = tasks
    assert o._get_tasks() is tasks


def test_orchestrator_is_done_completed():
    o = Orchestrator.__new__(Orchestrator)
    o._current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert o._is_done() is True


def test_orchestrator_is_done_failed():
    o = Orchestrator.__new__(Orchestrator)
    o._current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert o._is_done() is False


def test_orchestrator_is_done_pending():
    o = Orchestrator.__new__(Orchestrator)
    o._current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert o._is_done() is False


def test_orchestrator_is_done_no_tasks():
    o = Orchestrator.__new__(Orchestrator)
    o._current_tasks = []
    assert o._is_done() is True


def test_orchestrator_save_state(tmp_path):
    o = Orchestrator.__new__(Orchestrator)
    o.goal = "test"
    o.cycles = 1
    o._current_tasks = [TaskModel(id="1", description="task")]
    o._state_file = tmp_path / "state.json"
    o._save_state()
    assert o._state_file.exists()
    data = json.loads(o._state_file.read_text())
    assert data["goal"] == "test"


def test_orchestrator_load_state(tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({
        "goal": "loaded",
        "cycles": 2,
        "tasks": [{"id": "1", "description": "task"}],
    }))
    o = Orchestrator.__new__(Orchestrator)
    o._state_file = state_file
    o._load_state()
    assert o.goal == "loaded"
    assert o.cycles == 2


@pytest.mark.asyncio
async def test_llm_client_ollama_payload():
    from furrow.config import Provider
    settings = type("Settings", (), {
        "provider": Provider.OLLAMA,
        "model": "llama3",
        "ollama_base_url": "http://localhost:11434",
    })()
    client = LLMClient(settings=settings)

    mock_response = type("Resp", (), {
        "raise_for_status": AsyncMock(),
        "json": lambda self: {"message": {"content": "hi"}},
    })()
    client._ollama_client = type("Http", (), {
        "post": AsyncMock(return_value=mock_response),
    })()

    result = await client._complete_ollama("prompt", "sys", "llama3", 512)
    assert result == "hi"
