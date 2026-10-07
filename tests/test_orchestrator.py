import json
from pathlib import Path

import pytest

from furrow.config import Settings, TaskModel
from furrow.core.orchestrator import Orchestrator


class TestOrchestratorInit:
    def test_init_creates_tasks_list(self):
        orchestrator = Orchestrator("test goal")
        assert orchestrator.tasks == []

    def test_init_creates_cycles_zero(self):
        orchestrator = Orchestrator("test goal")
        assert orchestrator.cycles == 0

    def test_init_stores_goal(self):
        orchestrator = Orchestrator("my goal")
        assert orchestrator.goal == "my goal"

    def test_init_default_state_path(self):
        orchestrator = Orchestrator("test goal")
        assert orchestrator.state_path == Path(".furrow_state.json")

    def test_init_custom_state_path(self):
        orchestrator = Orchestrator("test goal", state_path="custom_state.json")
        assert orchestrator.state_path == Path("custom_state.json")


class TestOrchestratorIsDone:
    def test_is_done_returns_false_when_no_tasks(self):
        orchestrator = Orchestrator("test goal")
        assert orchestrator._is_done() is False

    def test_is_done_returns_true_when_all_completed(self):
        orchestrator = Orchestrator("test goal")
        orchestrator.tasks = [
            TaskModel(id="1", description="task1", status="completed"),
            TaskModel(id="2", description="task2", status="completed"),
        ]
        assert orchestrator._is_done() is True

    def test_is_done_returns_false_when_any_pending(self):
        orchestrator = Orchestrator("test goal")
        orchestrator.tasks = [
            TaskModel(id="1", description="task1", status="completed"),
            TaskModel(id="2", description="task2", status="pending"),
        ]
        assert orchestrator._is_done() is False

    def test_is_done_returns_false_when_any_failed(self):
        orchestrator = Orchestrator("test goal")
        orchestrator.tasks = [
            TaskModel(id="1", description="task1", status="completed"),
            TaskModel(id="2", description="task2", status="failed"),
        ]
        assert orchestrator._is_done() is False

    def test_is_done_returns_false_when_all_failed(self):
        orchestrator = Orchestrator("test goal")
        orchestrator.tasks = [
            TaskModel(id="1", description="task1", status="failed"),
            TaskModel(id="2", description="task2", status="failed"),
        ]
        assert orchestrator._is_done() is False

    def test_is_done_mixed_status_with_failed(self):
        orchestrator = Orchestrator("test goal")
        orchestrator.tasks = [
            TaskModel(id="1", description="task1", status="completed"),
            TaskModel(id="2", description="task2", status="failed"),
            TaskModel(id="3", description="task3", status="completed"),
        ]
        assert orchestrator._is_done() is False


class TestOrchestratorGetTasks:
    def test_get_tasks_returns_tasks(self):
        orchestrator = Orchestrator("test goal")
        orchestrator.tasks = [
            TaskModel(id="1", description="task1"),
            TaskModel(id="2", description="task2"),
        ]
        tasks = orchestrator._get_tasks()
        assert tasks == orchestrator.tasks
        assert len(tasks) == 2


class TestOrchestratorSaveLoadState:
    def test_save_state_writes_valid_json(self, tmp_path):
        state_file = tmp_path / "state.json"
        orchestrator = Orchestrator("test goal")
        orchestrator.goal = "updated goal"
        orchestrator.cycles = 3
        orchestrator.tasks = [
            TaskModel(id="1", description="task1", status="completed", result="done"),
        ]
        orchestrator.state_path = state_file
        orchestrator.save_state()

        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["goal"] == "updated goal"
        assert data["cycles"] == 3
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["id"] == "1"
        assert data["tasks"][0]["status"] == "completed"

    def test_save_state_uses_default_path(self, tmp_path, monkeypatch):
        state_file = tmp_path / ".furrow_state.json"
        monkeypatch.chdir(tmp_path)
        orchestrator = Orchestrator("test goal")
        orchestrator.cycles = 1
        orchestrator.tasks = [
            TaskModel(id="1", description="task1", status="completed"),
        ]
        orchestrator.save_state()
        assert state_file.exists()

    def test_load_state_returns_false_when_file_missing(self, tmp_path):
        state_file = tmp_path / "nonexistent.json"
        orchestrator = Orchestrator("test goal")
        result = orchestrator.load_state(path=state_file)
        assert result is False
        assert orchestrator.goal == "test goal"

    def test_load_state_loads_correctly(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "goal": "loaded goal",
            "cycles": 5,
            "tasks": [
                {
                    "id": "1",
                    "description": "task1",
                    "files": [],
                    "dependencies": [],
                    "status": "completed",
                    "result": "done",
                }
            ],
        }))
        orchestrator = Orchestrator("original goal")
        result = orchestrator.load_state(path=state_file)
        assert result is True
        assert orchestrator.goal == "loaded goal"
        assert orchestrator.cycles == 5
        assert len(orchestrator.tasks) == 1
        assert orchestrator.tasks[0].id == "1"
        assert orchestrator.tasks[0].status == "completed"
        assert orchestrator.tasks[0].result == "done"

    def test_load_state_returns_false_on_invalid_json(self, tmp_path):
        state_file = tmp_path / "bad.json"
        state_file.write_text("not valid json {{{")
        orchestrator = Orchestrator("test goal")
        result = orchestrator.load_state(path=state_file)
        assert result is False
