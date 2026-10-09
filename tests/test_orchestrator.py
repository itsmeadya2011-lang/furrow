from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class MockLLMClient:
    def __init__(self):
        self.settings = MagicMock()
        self.settings.max_cycles = 0
        self.settings.workspace = Path.cwd()
        self.planner_response = None
        self.tester_response = None
        self.complete = AsyncMock(side_effect=self._complete_side_effect)

    def _complete_side_effect(self, prompt, model=None, system=""):
        if "planning" in prompt.lower() or "plan" in prompt.lower():
            return self.planner_response or json.dumps({
                "tasks": [],
                "rationale": "no tasks"
            })
        return self.tester_response or json.dumps({
            "passed": True,
            "summary": "all good",
            "failures": []
        })


def test_original_and_working_goal():
    client = MockLLMClient()
    client.planner_response = json.dumps({"tasks": [], "rationale": "done"})
    orch = Orchestrator(goal="My goal", client=client)
    assert orch.original_goal == "My goal"
    assert orch.working_goal == "My goal"


def test_all_tasks_accumulates():
    client = MockLLMClient()
    client.planner_response = json.dumps({
        "tasks": [
            {"id": "1", "description": "task 1", "files": []},
            {"id": "2", "description": "task 2", "files": []},
        ],
        "rationale": "two tasks"
    })
    orch = Orchestrator(goal="test", client=client)
    assert orch._all_tasks == []
    task1 = TaskModel(id="1", description="task 1")
    task2 = TaskModel(id="2", description="task 2")
    orch._all_tasks.extend([task1, task2])
    assert len(orch._all_tasks) == 2


def test_is_done_no_tasks():
    client = MockLLMClient()
    orch = Orchestrator(goal="test", client=client)
    assert orch._is_done() is False


def test_is_done_all_completed():
    client = MockLLMClient()
    orch = Orchestrator(goal="test", client=client)
    task = TaskModel(id="1", description="task", status="completed")
    orch._all_tasks.append(task)
    orch._last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is True


def test_is_done_with_failed():
    client = MockLLMClient()
    orch = Orchestrator(goal="test", client=client)
    task = TaskModel(id="1", description="task", status="failed")
    orch._all_tasks.append(task)
    orch._last_test_result = TestResult(passed=False, summary="fail", failures=["err"])
    assert orch._is_done() is False


def test_max_cycles_enforced(tmp_path):
    client = MockLLMClient()
    client.settings.max_cycles = 2
    client.planner_response = json.dumps({"tasks": [], "rationale": "done"})
    state_file = tmp_path / "state.json"
    orch = Orchestrator(goal="test", client=client, state_path=state_file)
    messages = []
    orch._on_output = lambda msg, renderable=None: messages.append(msg)
    import asyncio
    asyncio.run(orch.run())
    assert orch.cycles == 2


def test_on_output_callback():
    client = MockLLMClient()
    client.planner_response = json.dumps({"tasks": [], "rationale": "done"})
    orch = Orchestrator(goal="test", client=client)
    messages = []
    orch._on_output = lambda msg, renderable=None: messages.append(msg)
    assert len(messages) == 0
    orch._print("hello")
    assert "hello" in messages


def test_state_save_load(tmp_path):
    client = MockLLMClient()
    state_file = tmp_path / "state.json"
    orch = Orchestrator(goal="test", client=client, state_path=state_file)
    task = TaskModel(id="1", description="task", status="completed")
    orch._all_tasks.append(task)
    orch._save_state()
    assert state_file.exists()

    orch2 = Orchestrator(goal="other", client=client, state_path=state_file)
    orch2._load_state()
    assert len(orch2._all_tasks) == 1
    assert orch2._all_tasks[0].description == "task"