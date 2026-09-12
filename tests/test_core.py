import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


# --- Existing tests ---

def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# --- Settings ---

def test_settings_persist_path():
    s = Settings()
    assert s.persist_path == Path(".furrow/state.json")


# --- Orchestrator task tracking ---

def test_orchestrator_get_tasks_returns_stored():
    o = Orchestrator(goal="test")
    assert o._get_tasks() == []
    o._tasks = [TaskModel(id="1", description="task1", status="completed")]
    assert len(o._get_tasks()) == 1
    assert o._get_tasks()[0].id == "1"


def test_orchestrator_is_done_all_completed():
    o = Orchestrator(goal="test")
    o._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert o._is_done() is True


def test_orchestrator_is_done_with_failed():
    o = Orchestrator(goal="test")
    o._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert o._is_done() is False


def test_orchestrator_is_done_pending():
    o = Orchestrator(goal="test")
    o._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert o._is_done() is False


def test_orchestrator_is_done_empty():
    o = Orchestrator(goal="test")
    assert o._is_done() is True  # No tasks = done


# --- State persistence ---

def test_save_and_load_state():
    o = Orchestrator(goal="original goal")
    o._tasks = [TaskModel(id="1", description="task1", status="completed")]

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.json"
        o.client.settings.persist_path = state_path

        asyncio.run(o.save_state())
        assert state_path.exists()

        data = json.loads(state_path.read_text())
        assert data["goal"] == "original goal"
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["id"] == "1"

        # Load into a new orchestrator
        o2 = Orchestrator(goal="different")
        o2.client.settings.persist_path = state_path
        loaded = asyncio.run(o2.load_state())
        assert loaded is True
        assert o2.goal == "original goal"
        assert len(o2._tasks) == 1
        assert o2._tasks[0].id == "1"


def test_load_state_no_file():
    o = Orchestrator(goal="test")
    with tempfile.TemporaryDirectory() as tmpdir:
        o.client.settings.persist_path = Path(tmpdir) / "nonexistent.json"
        loaded = asyncio.run(o.load_state())
        assert loaded is False


# --- Worker code writing ---

def test_worker_parses_code_blocks():
    from furrow.agents.worker import WorkerAgent

    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value="""
Here's the implementation:

```python
# filename: src/app.py
print("hello world")
```

```javascript
// filename: src/app.js
console.log("hello");
```
""")
    mock_client.settings.worker_model = "test-model"
    mock_client.write_file = AsyncMock()

    task = TaskModel(id="1", description="create app files")
    worker = WorkerAgent(task=task, client=mock_client)
    result = asyncio.run(worker.run())

    assert mock_client.write_file.call_count == 2
    assert "src/app.py" in result
    assert "src/app.js" in result
    # Verify the filename annotation line is stripped from the written code
    calls = mock_client.write_file.call_args_list
    py_call = next(c for c in calls if c.args[0] == "src/app.py")
    assert "# filename:" not in py_call.args[1]
    js_call = next(c for c in calls if c.args[0] == "src/app.js")
    assert "// filename:" not in js_call.args[1]


def test_worker_uses_task_files_when_no_filename():
    from furrow.agents.worker import WorkerAgent

    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value="""
```python
code here
```
""")
    mock_client.settings.worker_model = "test-model"
    mock_client.write_file = AsyncMock()

    task = TaskModel(id="1", description="create file", files=["src/main.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    result = asyncio.run(worker.run())

    assert mock_client.write_file.call_count == 1
    mock_client.write_file.assert_called_with("src/main.py", "code here\n")


def test_worker_no_blocks():
    from furrow.agents.worker import WorkerAgent

    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value="No code blocks here.")
    mock_client.settings.worker_model = "test-model"
    mock_client.write_file = AsyncMock()

    task = TaskModel(id="1", description="just talk")
    worker = WorkerAgent(task=task, client=mock_client)
    result = asyncio.run(worker.run())

    assert mock_client.write_file.call_count == 0
    assert "0 files" in result


# --- Planner retry ---

def test_planner_retries_on_failure():
    from furrow.agents.planner import PlannerAgent

    mock_client = AsyncMock()
    # Fail twice with bad JSON, then succeed
    mock_client.complete = AsyncMock(side_effect=[
        "not json",
        "still not json",
        '{"tasks": [{"id": "1", "description": "done"}], "rationale": "ok"}',
    ])
    mock_client.settings.planner_model = "test-model"

    planner = PlannerAgent(client=mock_client)
    plan = asyncio.run(planner.plan("test goal"))

    assert mock_client.complete.call_count == 3
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"


def test_planner_fails_after_3_attempts():
    from furrow.agents.planner import PlannerAgent

    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value="not json at all")
    mock_client.settings.planner_model = "test-model"

    planner = PlannerAgent(client=mock_client)
    with pytest.raises(ValueError, match="Failed to parse plan after 3 attempts"):
        asyncio.run(planner.plan("test goal"))

    assert mock_client.complete.call_count == 3


# --- Broadcast ---

def test_orchestrator_emit_with_broadcast():
    o = Orchestrator(goal="test")
    messages = []

    async def broadcast(msg):
        messages.append(msg)

    o.broadcast = broadcast
    asyncio.run(o._emit("hello"))
    assert "hello" in messages


def test_orchestrator_emit_without_broadcast():
    o = Orchestrator(goal="test")
    o.broadcast = None
    # Should not raise
    asyncio.run(o._emit("hello"))