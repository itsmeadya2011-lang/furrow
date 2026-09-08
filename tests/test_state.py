import json
from pathlib import Path

import pytest

from furrow.config import TaskModel
from furrow.core.orchestrator import Orchestrator


class TestStatePersistence:
    @pytest.mark.asyncio
    async def test_save_and_load_state(self, tmp_path):
        state_file = tmp_path / "state.json"
        o = Orchestrator.__new__(Orchestrator)
        o.goal = "do stuff"
        o.cycles = 3
        o._current_tasks = [
            TaskModel(id="1", description="task1", status="completed"),
            TaskModel(id="2", description="task2", status="completed"),
        ]
        o._state_file = state_file

        o._save_state()

        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["goal"] == "do stuff"
        assert data["cycles"] == 3
        assert len(data["tasks"]) == 2
        assert data["tasks"][0]["id"] == "1"

    @pytest.mark.asyncio
    async def test_load_state(self, tmp_path):
        state_file = tmp_path / "state.json"
        state = {
            "goal": "loaded goal",
            "cycles": 7,
            "tasks": [
                {"id": "1", "description": "task1", "files": [], "dependencies": [], "status": "pending", "result": None},
            ],
        }
        state_file.write_text(json.dumps(state))

        o = Orchestrator.__new__(Orchestrator)
        o.goal = "original"
        o.cycles = 0
        o._current_tasks = []
        o._state_file = state_file

        o._load_state()

        assert o.goal == "loaded goal"
        assert o.cycles == 7
        assert len(o._current_tasks) == 1
        assert o._current_tasks[0].id == "1"

    @pytest.mark.asyncio
    async def test_load_missing_state(self, tmp_path):
        o = Orchestrator.__new__(Orchestrator)
        o._state_file = tmp_path / "missing.json"
        o._load_state()
        assert True

    @pytest.mark.asyncio
    async def test_load_corrupted_state(self, tmp_path):
        state_file = tmp_path / "bad.json"
        state_file.write_text("not json")

        o = Orchestrator.__new__(Orchestrator)
        o._state_file = state_file
        o._load_state()
        assert True

    @pytest.mark.asyncio
    async def test_round_trip_state(self, tmp_path):
        state_file = tmp_path / "round_trip.json"
        o = Orchestrator.__new__(Orchestrator)
        o.goal = "round trip"
        o.cycles = 5
        o._current_tasks = [
            TaskModel(id="1", description="a", status="completed", result="done"),
            TaskModel(id="2", description="b", status="pending"),
        ]
        o._state_file = state_file

        o._save_state()

        o2 = Orchestrator.__new__(Orchestrator)
        o2._state_file = state_file
        o2._load_state()

        assert o2.goal == "round trip"
        assert o2.cycles == 5
        assert len(o2._current_tasks) == 2
        assert o2._current_tasks[0].result == "done"
        assert o2._current_tasks[1].status == "pending"
