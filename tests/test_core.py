import asyncio
from pathlib import Path

import pytest
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_done_with_no_tasks():
    o = Orchestrator(goal="test")
    assert o._is_done() is True


def test_orchestrator_done_with_completed_tasks():
    o = Orchestrator(goal="test")
    o._current_plan = Plan(
        tasks=[TaskModel(id="1", description="do thing", status="completed")],
        rationale="ok",
    )
    assert o._is_done() is True


def test_orchestrator_not_done_with_failed_tasks():
    o = Orchestrator(goal="test")
    o._current_plan = Plan(
        tasks=[TaskModel(id="1", description="do thing", status="failed")],
        rationale="ok",
    )
    assert o._is_done() is False


def test_orchestrator_not_done_with_pending_tasks():
    o = Orchestrator(goal="test")
    o._current_plan = Plan(
        tasks=[TaskModel(id="1", description="do thing", status="pending")],
        rationale="ok",
    )
    assert o._is_done() is False


@pytest.mark.asyncio
async def test_worker_writes_files(tmp_path: Path):
    task = TaskModel(id="1", description="create a hello world script", files=["hello.py"])
    worker = WorkerAgent(task=task, workspace=tmp_path)
    response = """Here is the file:

FILE: hello.py
```python
print("hello")
```
"""
    written = worker._write_files_from_response(response)
    assert written == ["hello.py"]
    assert (tmp_path / "hello.py").read_text() == 'print("hello")\n'


@pytest.mark.asyncio
async def test_worker_no_files_written(tmp_path: Path):
    task = TaskModel(id="1", description="do nothing", files=[])
    worker = WorkerAgent(task=task, workspace=tmp_path)
    response = "Just some text without file markers."
    written = worker._write_files_from_response(response)
    assert written == []
