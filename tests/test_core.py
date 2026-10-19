import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.worker import FileOperation, WorkerAgent, WorkerResult
from furrow.config import Plan, Settings, TaskModel, TestResult, configure_logging
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_file_operation_model():
    op = FileOperation(path="src/foo.py", operation="create", content="print('hi')")
    assert op.path == "src/foo.py"
    assert op.operation == "create"
    assert op.content == "print('hi')"


def test_worker_result_model():
    r = WorkerResult(
        task_id="1",
        status="completed",
        files_modified=["src/foo.py"],
        summary="Created foo.py",
    )
    assert r.task_id == "1"
    assert r.status == "completed"
    assert r.files_modified == ["src/foo.py"]
    assert r.summary == "Created foo.py"


# ---------------------------------------------------------------------------
# configure_logging tests
# ---------------------------------------------------------------------------

def test_configure_logging_runs():
    # Should not raise
    configure_logging("INFO")


# ---------------------------------------------------------------------------
# WorkerAgent tests
# ---------------------------------------------------------------------------

class FakeLLMClient:
    """A fake LLMClient that returns canned JSON responses."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.read_file_calls: list[str] = []
        self.write_file_calls: list[tuple[str, str]] = []
        self.settings = MagicMock()
        self.settings.worker_model = "test-model"
        self.settings.planner_model = "test-planner"
        self.settings.tester_model = "test-tester"

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        return self.response_text

    async def read_file(self, path: str | Path) -> str:
        self.read_file_calls.append(str(path))
        return "# existing content\n"

    async def write_file(self, path: str | Path, content: str) -> None:
        self.write_file_calls.append((str(path), content))

    def list_files(self, directory: str | Path) -> list[str]:
        return []


@pytest.mark.asyncio
async def test_worker_agent_modifies_file(tmp_path):
    """Worker should read existing files, call LLM, and write modified content."""
    fake_client = FakeLLMClient(
        json.dumps({
            "operations": [
                {"path": "src/foo.py", "operation": "modify", "content": "# new content\n"}
            ],
            "summary": "Updated foo.py",
        })
    )
    task = TaskModel(id="1", description="Update foo.py", files=["src/foo.py"])
    worker = WorkerAgent(task=task, client=fake_client, workspace=tmp_path)

    # Pre-create the file so read_file finds it
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "foo.py").write_text("# existing content\n")

    result = await worker.run()

    assert result.status == "completed"
    assert "src/foo.py" in result.files_modified
    assert "Updated foo.py" in result.summary
    assert len(fake_client.write_file_calls) == 1
    assert fake_client.write_file_calls[0][1] == "# new content\n"


@pytest.mark.asyncio
async def test_worker_agent_creates_new_file(tmp_path):
    """Worker should be able to create new files."""
    fake_client = FakeLLMClient(
        json.dumps({
            "operations": [
                {"path": "src/bar.py", "operation": "create", "content": "print('hello')\n"}
            ],
            "summary": "Created bar.py",
        })
    )
    task = TaskModel(id="2", description="Create bar.py", files=["src/bar.py"])
    worker = WorkerAgent(task=task, client=fake_client, workspace=tmp_path)

    result = await worker.run()

    assert result.status == "completed"
    assert "src/bar.py" in result.files_modified
    assert len(fake_client.write_file_calls) == 1


@pytest.mark.asyncio
async def test_worker_agent_invalid_json(tmp_path):
    """Worker should gracefully handle invalid JSON from LLM."""
    fake_client = FakeLLMClient("not json at all")
    task = TaskModel(id="3", description="Do something", files=[])
    worker = WorkerAgent(task=task, client=fake_client, workspace=tmp_path)

    result = await worker.run()

    assert result.status == "failed"
    assert result.error is not None


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

def test_orchestrator_initial_state():
    orch = Orchestrator(goal="Test goal")
    assert orch.goal == "Test goal"
    assert orch.cycles == 0
    assert orch.tasks == []


def test_orchestrator_is_done_no_tasks():
    """With no tasks, _is_done should return False."""
    orch = Orchestrator(goal="Test goal")
    assert orch._is_done() is False


def test_orchestrator_is_done_all_completed():
    """When all tasks completed, _is_done should return True."""
    orch = Orchestrator(goal="Test goal")
    t1 = TaskModel(id="1", description="one", status="completed")
    t2 = TaskModel(id="2", description="two", status="completed")
    orch.tasks = [t1, t2]
    assert orch._is_done() is True


def test_orchestrator_is_done_one_failed():
    """When any task failed, _is_done should return False."""
    orch = Orchestrator(goal="Test goal")
    t1 = TaskModel(id="1", description="one", status="completed")
    t2 = TaskModel(id="2", description="two", status="failed")
    orch.tasks = [t1, t2]
    assert orch._is_done() is False


def test_orchestrator_max_cycles():
    """max_cycles=0 means unlimited; positive value limits cycles."""
    orch = Orchestrator(goal="Test goal")
    # max_cycles from default settings is 0 (unlimited)
    assert orch.cfg.max_cycles == 0

    # Explicitly set max_cycles via settings
    settings = Settings(max_cycles=2)
    orch2 = Orchestrator(goal="Test goal", settings=settings)
    assert orch2.cfg.max_cycles == 2


def test_orchestrator_get_tasks_returns_tracked_tasks():
    """_get_tasks should return the tasks tracked by the orchestrator."""
    orch = Orchestrator(goal="Test goal")
    t1 = TaskModel(id="1", description="one")
    t2 = TaskModel(id="2", description="two")
    orch.tasks = [t1, t2]
    assert orch._get_tasks() == [t1, t2]


@pytest.mark.asyncio
async def test_orchestrator_cycle_no_planned_tasks():
    """When planner returns no tasks, cycle completes without error."""
    fake_client = FakeLLMClient(json.dumps({"tasks": [], "rationale": "done"}))
    orch = Orchestrator(goal="Test goal", client=fake_client)
    await orch._cycle()
    assert orch.tasks == []


# ---------------------------------------------------------------------------
# Settings tests
# ---------------------------------------------------------------------------

def test_settings_defaults():
    s = Settings()
    assert s.provider.value == "anthropic"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_settings_max_cycles_override():
    s = Settings(max_cycles=10)
    assert s.max_cycles == 10
