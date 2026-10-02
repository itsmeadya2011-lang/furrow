"""Tests for the persistent state module."""

from __future__ import annotations

from furrow.config import TaskModel, TestResult
from furrow.core.state import clear_state, load_state, save_state


class _FakeOrchestrator:
    """Minimal stand-in that exposes the attributes ``save_state`` reads."""

    def __init__(self, goal, cycles, tasks, last_test_result=None):
        self.goal = goal
        self.cycles = cycles
        self.tasks = tasks
        self.last_test_result = last_test_result
        self.max_cycles = 0
        self.max_parallel_tasks = 5


class TestSaveLoadState:
    def test_save_and_load_roundtrip(self, tmp_path):
        state_path = tmp_path / "furrow-state.json"
        tasks = [
            TaskModel(id="1", description="create file", status="completed", result="done"),
            TaskModel(id="2", description="test file", status="pending"),
        ]
        orch = _FakeOrchestrator(
            goal="Add feature X",
            cycles=3,
            tasks=tasks,
            last_test_result=TestResult(passed=True, summary="all green"),
        )
        save_state(orch, state_path)

        data = load_state(state_path)
        assert data is not None
        assert data["goal"] == "Add feature X"
        assert data["cycles"] == 3
        assert len(data["tasks"]) == 2
        assert data["tasks"][0]["status"] == "completed"
        assert data["last_test_result"]["passed"] is True

    def test_load_state_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent.json"
        assert load_state(missing) is None

    def test_load_state_corrupt_file(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json{", encoding="utf-8")
        assert load_state(bad) is None

    def test_load_state_without_test_result(self, tmp_path):
        state_path = tmp_path / "state.json"
        tasks = [TaskModel(id="1", description="t")]
        orch = _FakeOrchestrator(goal="g", cycles=1, tasks=tasks)
        save_state(orch, state_path)

        data = load_state(state_path)
        assert "last_test_result" not in data

    def test_clear_state(self, tmp_path):
        state_path = tmp_path / "state.json"
        state_path.write_text("{}", encoding="utf-8")
        clear_state(state_path)
        assert not state_path.exists()

    def test_clear_state_nonexistent(self, tmp_path):
        """Clearing a non-existent file should be a no-op."""
        clear_state(tmp_path / "nope.json")  # should not raise


class TestOrchestratorStateIntegration:
    """Integration test: Orchestrator saves and loads its own state."""

    def test_orchestrator_state_file_param(self, tmp_path):
        from furrow.core.orchestrator import Orchestrator

        state_path = tmp_path / "orch-state.json"
        tasks = [TaskModel(id="1", description="t", status="completed")]
        orch = Orchestrator(
            goal="test goal",
            state_file=state_path,
        )
        orch.tasks = tasks
        orch.cycles = 5
        import asyncio

        asyncio.run(orch._save_persisted_state())

        loaded = load_state(state_path)
        assert loaded["goal"] == "test goal"
        assert loaded["cycles"] == 5
        assert loaded["tasks"][0]["status"] == "completed"
