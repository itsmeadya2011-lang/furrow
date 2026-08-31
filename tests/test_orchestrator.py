from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _make_client():
    client = AsyncMock()
    client.settings = Settings()
    return client


class TestOrchestratorInit:
    def test_default_initialization(self):
        orch = Orchestrator("build a thing")
        assert orch.goal == "build a thing"
        assert orch.cycles == 0
        assert orch.plan is None
        assert orch.all_tasks == []
        assert orch.history == []

    def test_with_custom_client(self):
        client = AsyncMock()
        orch = Orchestrator("do stuff", client=client)
        assert orch.client is client


class TestIsDone:
    def test_no_plan_returns_true(self):
        orch = Orchestrator("goal")
        orch.plan = None
        assert orch._is_done() is True

    def test_empty_tasks_returns_true(self):
        orch = Orchestrator("goal")
        orch.plan = Plan(tasks=[], rationale="none")
        assert orch._is_done() is True

    def test_all_completed_returns_true(self):
        orch = Orchestrator("goal")
        orch.plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="completed"),
            ],
            rationale="ok",
        )
        assert orch._is_done() is True

    def test_some_pending_returns_false(self):
        orch = Orchestrator("goal")
        orch.plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="pending"),
            ],
            rationale="ok",
        )
        assert orch._is_done() is False

    def test_any_failed_returns_false(self):
        orch = Orchestrator("goal")
        orch.plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="failed"),
            ],
            rationale="ok",
        )
        assert orch._is_done() is False

    def test_all_failed_returns_false(self):
        orch = Orchestrator("goal")
        orch.plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="failed"),
                TaskModel(id="2", description="b", status="failed"),
            ],
            rationale="ok",
        )
        assert orch._is_done() is False


class TestGetTasks:
    def test_returns_plan_tasks(self):
        orch = Orchestrator("goal")
        tasks = [
            TaskModel(id="1", description="a"),
            TaskModel(id="2", description="b"),
        ]
        orch.plan = Plan(tasks=tasks, rationale="ok")
        result = orch._get_tasks()
        assert result == tasks

    def test_returns_empty_when_no_plan(self):
        orch = Orchestrator("goal")
        assert orch._get_tasks() == []


class TestStateSaveLoad:
    def test_save_state_creates_file(self, tmp_path):
        with patch("furrow.config.settings") as mock_settings:
            mock_settings.workspace = tmp_path
            orch = Orchestrator("build app")
            orch.cycles = 3
            orch.plan = Plan(
                tasks=[TaskModel(id="1", description="x")], rationale="ok"
            )
            orch.history = [{"cycle": 1, "status": "passed"}]
            orch._save_state()

            state_file = tmp_path / ".furrow_state.json"
            assert state_file.exists()

            data = json.loads(state_file.read_text())
            assert data["goal"] == "build app"
            assert data["cycles"] == 3
            assert len(data["plan"]["tasks"]) == 1
            assert len(data["history"]) == 1

    def test_load_state_restores_state(self, tmp_path):
        with patch("furrow.config.settings") as mock_settings:
            mock_settings.workspace = tmp_path
            orch = Orchestrator("original goal")
            state = {
                "goal": "loaded goal",
                "cycles": 5,
                "plan": {
                    "tasks": [{"id": "1", "description": "t", "files": [], "dependencies": [], "status": "completed", "result": None}],
                    "rationale": "loaded",
                },
                "all_tasks": [
                    {"id": "1", "description": "t", "files": [], "dependencies": [], "status": "completed", "result": None}
                ],
                "history": [{"cycle": 1, "status": "passed"}],
            }
            (tmp_path / ".furrow_state.json").write_text(json.dumps(state))

            orch._load_state()

            assert orch.goal == "loaded goal"
            assert orch.cycles == 5
            assert orch.plan is not None
            assert orch.plan.tasks[0].id == "1"
            assert orch.all_tasks[0].id == "1"
            assert len(orch.history) == 1

    def test_load_state_missing_file_does_nothing(self):
        with patch("furrow.config.settings") as mock_settings:
            mock_settings.workspace = Path("/nonexistent")
            orch = Orchestrator("goal")
            orch._load_state()  # should not raise
            assert orch.goal == "goal"

    def test_load_state_corrupt_file_does_nothing(self, tmp_path):
        with patch("furrow.config.settings") as mock_settings:
            mock_settings.workspace = tmp_path
            (tmp_path / ".furrow_state.json").write_text("not json")
            orch = Orchestrator("goal")
            orch._load_state()  # should not raise
            assert orch.goal == "goal"


class TestMaxCycles:
    @pytest.mark.asyncio
    async def test_run_stops_at_max_cycles(self, tmp_path):
        client = _make_client()
        with patch("furrow.config.settings") as mock_settings:
            mock_settings.workspace = tmp_path
            mock_settings.max_cycles = 2
            mock_settings.max_parallel_tasks = 5

            task = TaskModel(id="1", description="t")
            failing_plan = Plan(tasks=[task], rationale="ok")

            with patch.object(
                PlannerAgent, "plan", new_callable=AsyncMock
            ) as mock_plan:
                mock_plan.return_value = failing_plan
                with patch.object(
                    WorkerAgent, "run", new_callable=AsyncMock
                ) as mock_worker:
                    mock_worker.return_value = {"success": False, "summary": "fail"}
                    with patch.object(
                        TesterAgent, "run", new_callable=AsyncMock
                    ) as mock_tester:
                        mock_tester.return_value = TestResult(
                            passed=False, summary="fail", failures=["fail"]
                        )
                        orch = Orchestrator("goal", client=client)
                        orch._load_state = MagicMock()
                        orch._save_state = MagicMock()
                        with patch("rich.console.Console.print"):
                            with patch("rich.status.Status"):
                                await orch.run()

                        assert orch.cycles == 2
