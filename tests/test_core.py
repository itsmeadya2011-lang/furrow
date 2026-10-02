from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import CompletionResult, LLMClient, ToolCall

# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------

def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="t1", description="implement X")
    assert t.id == "t1"
    assert t.description == "implement X"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_with_deps():
    t = TaskModel(
        id="t2",
        description="build Y",
        files=["src/y.py"],
        dependencies=["t1"],
        status="completed",
        result="done",
    )
    assert t.dependencies == ["t1"]
    assert t.status == "completed"


def test_plan_from_llm_json():
    data = {
        "tasks": [
            {
                "id": "1",
                "description": "Add auth",
                "files": ["src/auth.py"],
                "dependencies": [],
            }
        ],
        "rationale": "Need auth module",
    }
    plan = Plan(**data)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.rationale == "Need auth module"


# ---------------------------------------------------------------------------
# LLMClient data structures
# ---------------------------------------------------------------------------

def test_tool_call_creation():
    tc = ToolCall(
        id="call_1",
        type="tool_use",
        name="read_file",
        input={"path": "test.txt"},
    )
    assert tc.id == "call_1"
    assert tc.name == "read_file"
    assert tc.input == {"path": "test.txt"}


def test_completion_result_defaults():
    result = CompletionResult(text="done", tool_calls=[])
    assert result.text == "done"
    assert result.tool_calls == []


def test_completion_result_with_tools():
    tc = ToolCall(id="1", type="tool_use", name="write_file", input={})
    result = CompletionResult(text="working", tool_calls=[tc])
    assert len(result.tool_calls) == 1


# ---------------------------------------------------------------------------
# LLMClient execute_tool tests
# ---------------------------------------------------------------------------

@pytest.fixture
def file_client(tmp_path):
    client = LLMClient.__new__(LLMClient)
    client.settings = MagicMock()
    client.settings.workspace = tmp_path
    return client


async def test_execute_read_file(file_client, tmp_path):
    filepath = tmp_path / "hello.txt"
    filepath.write_text("hello world")
    tc = ToolCall(id="1", type="tool_use", name="read_file", input={"path": str(filepath)})
    result = await file_client.execute_tool(tc)
    assert "hello world" in result


async def test_execute_write_file(file_client, tmp_path):
    filepath = tmp_path / "output.txt"
    tc = ToolCall(
        id="1",
        type="tool_use",
        name="write_file",
        input={"path": str(filepath), "content": "written content"},
    )
    result = await file_client.execute_tool(tc)
    assert "Wrote" in result
    assert filepath.read_text() == "written content"


async def test_execute_list_files(file_client, tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.py").write_text("b")
    tc = ToolCall(id="1", type="tool_use", name="list_files", input={"directory": str(tmp_path)})
    result = await file_client.execute_tool(tc)
    files = json.loads(result)
    assert "a.txt" in files
    assert "b.py" in files


async def test_execute_edit_file(file_client, tmp_path):
    filepath = tmp_path / "edit.txt"
    filepath.write_text("old content here")
    tc = ToolCall(
        id="1",
        type="tool_use",
        name="edit_file",
        input={"path": str(filepath), "old_string": "old", "new_string": "new"},
    )
    result = await file_client.execute_tool(tc)
    assert "Updated" in result
    assert filepath.read_text() == "new content here"


async def test_execute_edit_file_not_found(file_client, tmp_path):
    filepath = tmp_path / "missing.txt"
    filepath.write_text("content")
    tc = ToolCall(
        id="1",
        type="tool_use",
        name="edit_file",
        input={"path": str(filepath), "old_string": "notfound", "new_string": "x"},
    )
    with pytest.raises(ValueError, match="not found"):
        await file_client.execute_tool(tc)


async def test_execute_unknown_tool(file_client):
    tc = ToolCall(id="1", type="tool_use", name="nonexistent_tool", input={})
    with pytest.raises(ValueError, match="Unknown tool"):
        await file_client.execute_tool(tc)


# ---------------------------------------------------------------------------
# WorkerAgent tests
# ---------------------------------------------------------------------------

async def test_worker_format_tool_results():
    result = CompletionResult(
        text="I read the file.",
        tool_calls=[
            ToolCall(
                id="call_1",
                type="tool_use",
                name="read_file",
                input={"path": "test.py"},
            )
        ],
    )
    formatted = WorkerAgent._format_tool_results(result, ["file contents here"])
    assert "read_file" in formatted
    assert "file contents here" in formatted
    assert "Do not repeat" in formatted


async def test_worker_format_tool_results_no_text():
    result = CompletionResult(
        text="",
        tool_calls=[
            ToolCall(
                id="call_1",
                type="tool_use",
                name="write_file",
                input={"path": "out.py", "content": "..."},
            )
        ],
    )
    formatted = WorkerAgent._format_tool_results(result, ["Wrote content to out.py"])
    assert "write_file" in formatted
    assert "Wrote content to out.py" in formatted


async def test_worker_max_rounds():
    """Worker should return a max-rounds message when LLM keeps calling tools."""
    client = MagicMock()
    tool_call = ToolCall(id="1", type="tool_use", name="list_files", input={"directory": "."})
    client.complete_with_tools = AsyncMock(
        return_value=CompletionResult(text="looking...", tool_calls=[tool_call])
    )
    client.execute_tool = AsyncMock(return_value="file1.py")
    client.settings = MagicMock()
    client.settings.worker_model = "test-model"

    task = TaskModel(id="1", description="test task")
    # Temporarily reduce MAX_ROUNDS for test speed
    original_max = WorkerAgent.MAX_ROUNDS
    WorkerAgent.MAX_ROUNDS = 3
    try:
        worker = WorkerAgent(task=task, client=client)
        result = await worker.run()
        assert "max rounds" in result.lower()
        assert client.complete_with_tools.call_count == 3
    finally:
        WorkerAgent.MAX_ROUNDS = original_max


async def test_worker_returns_text_when_no_tool_calls():
    """Worker should return LLM text directly when no tool calls are made."""
    client = MagicMock()
    client.complete_with_tools = AsyncMock(
        return_value=CompletionResult(text="Task done!", tool_calls=[])
    )
    client.settings = MagicMock()
    client.settings.worker_model = "test-model"

    task = TaskModel(id="1", description="test task")
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert result == "Task done!"
    assert client.complete_with_tools.call_count == 1


async def test_worker_handles_tool_error():
    """Worker should continue when a tool call raises an error."""
    client = MagicMock()
    tool_call = ToolCall(id="1", type="tool_use", name="read_file", input={"path": "x.py"})
    # First call: tool calls, second call: no tool calls
    client.complete_with_tools = AsyncMock(
        return_value=CompletionResult(text="", tool_calls=[]),
        side_effect=[
            CompletionResult(text="reading", tool_calls=[tool_call]),
            CompletionResult(text="Error was handled, task complete.", tool_calls=[]),
        ],
    )
    client.execute_tool = AsyncMock(side_effect=RuntimeError("Disk error"))
    client.settings = MagicMock()
    client.settings.worker_model = "test-model"

    task = TaskModel(id="1", description="test task")
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert "Error was handled" in result


# ---------------------------------------------------------------------------
# Orchestrator topological_batches tests
# ---------------------------------------------------------------------------

def test_topological_batches_no_deps():
    tasks = [
        TaskModel(id="1", description="A"),
        TaskModel(id="2", description="B"),
        TaskModel(id="3", description="C"),
    ]
    batches = Orchestrator._topological_batches(tasks)
    assert len(batches) == 1
    assert len(batches[0]) == 3


def test_topological_batches_with_deps():
    tasks = [
        TaskModel(id="1", description="A"),
        TaskModel(id="2", description="B", dependencies=["1"]),
        TaskModel(id="3", description="C", dependencies=["2"]),
    ]
    batches = Orchestrator._topological_batches(tasks)
    assert len(batches) == 3
    assert batches[0][0].id == "1"
    assert batches[1][0].id == "2"
    assert batches[2][0].id == "3"


def test_topological_batches_partial_deps():
    """Task with deps in earlier batch, independent task in first batch."""
    tasks = [
        TaskModel(id="1", description="A"),
        TaskModel(id="2", description="B", dependencies=["1"]),
        TaskModel(id="3", description="C"),
    ]
    batches = Orchestrator._topological_batches(tasks)
    assert len(batches) == 2
    batch0_ids = {t.id for t in batches[0]}
    batch1_ids = {t.id for t in batches[1]}
    assert batch0_ids == {"1", "3"}
    assert batch1_ids == {"2"}


def test_topological_batches_circular_dep_fallback():
    """Circular dependencies should fall back to including all remaining."""
    tasks = [
        TaskModel(id="1", description="A", dependencies=["2"]),
        TaskModel(id="2", description="B", dependencies=["1"]),
    ]
    batches = Orchestrator._topological_batches(tasks)
    # First iteration: neither dep is satisfied, so fallback to all remaining
    assert len(batches) == 1
    assert len(batches[0]) == 2


def test_topological_batches_empty():
    batches = Orchestrator._topological_batches([])
    assert batches == []


# ---------------------------------------------------------------------------
# Orchestrator _is_done tests
# ---------------------------------------------------------------------------

def test_orchestrator_is_done_no_tasks():
    orch = Orchestrator.__new__(Orchestrator)
    orch.all_tasks = []
    assert orch._is_done() is True


def test_orchestrator_is_done_all_completed():
    orch = Orchestrator.__new__(Orchestrator)
    orch.all_tasks = [
        TaskModel(id="1", description="A", status="completed"),
        TaskModel(id="2", description="B", status="completed"),
    ]
    assert orch._is_done() is True


def test_orchestrator_is_done_with_failed():
    orch = Orchestrator.__new__(Orchestrator)
    orch.all_tasks = [
        TaskModel(id="1", description="A", status="completed"),
        TaskModel(id="2", description="B", status="failed"),
    ]
    assert orch._is_done() is False


def test_orchestrator_is_done_partial():
    orch = Orchestrator.__new__(Orchestrator)
    orch.all_tasks = [
        TaskModel(id="1", description="A", status="completed"),
        TaskModel(id="2", description="B", status="pending"),
    ]
    assert orch._is_done() is False


def test_orchestrator_get_tasks_returns_all_tasks():
    orch = Orchestrator.__new__(Orchestrator)
    tasks = [TaskModel(id="1", description="A")]
    orch.all_tasks = tasks
    assert orch._get_tasks() is tasks


# ---------------------------------------------------------------------------
# Orchestrator max_cycles test
# ---------------------------------------------------------------------------

async def test_orchestrator_max_cycles_async():
    from rich.console import Console

    orch = Orchestrator.__new__(Orchestrator)
    orch.goal = "test"
    orch.cycles = 0
    orch.all_tasks = []
    orch.client = MagicMock()
    orch.planner = MagicMock()
    orch.console = Console()

    mock_settings = MagicMock()
    mock_settings.max_cycles = 1
    mock_settings.max_parallel_tasks = 5
    mock_settings.workspace = Path(".")
    orch.settings = mock_settings

    async def fake_cycle():
        orch.all_tasks = [TaskModel(id="1", description="A", status="pending")]

    orch._cycle = fake_cycle
    await orch.run()
    # With max_cycles=1, should stop after the first cycle
    assert orch.cycles == 1
