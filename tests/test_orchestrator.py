import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Settings, TaskModel
from furrow.core.orchestrator import Orchestrator


class MockLLMClient:
    def __init__(self):
        self.settings = Settings()

    async def complete(self, prompt: str, model: str | None = None) -> str:
        return '{"tasks": [{"id": "1", "description": "task 1", "files": [], "dependencies": []}], "rationale": "test"}'


class MockPlannerAgent:
    def __init__(self, client=None, settings=None):
        pass

    async def plan(self, goal: str, previous_tasks=None):
        from furrow.config import Plan
        return Plan(
            tasks=[TaskModel(id="1", description="task 1", files=[], dependencies=[])],
            rationale="test",
        )


class MockWorkerAgent:
    def __init__(self, task, client=None, settings=None):
        self.task = task

    async def run(self):
        return "Done"


class MockTesterAgent:
    def __init__(self, client=None, settings=None):
        pass

    async def run(self, goal: str, tasks):
        from furrow.config import TestResult
        return TestResult(passed=True, summary="All passed", failures=[])


@pytest.mark.asyncio
async def test_orchestrator_initialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / ".furrow_state.json"
        client = MockLLMClient()
        config = Settings(workspace=tmpdir)

        orch = Orchestrator(
            goal="test goal",
            client=client,
            config=config,
            state_file=state_file,
        )

        assert orch.goal == "test goal"
        assert orch.original_goal == "test goal"
        assert orch.cycles == 0
        assert orch.all_tasks == []


@pytest.mark.asyncio
async def test_orchestrator_load_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / ".furrow_state.json"
        state_data = {
            "cycles": 2,
            "goal": "updated goal",
            "original_goal": "original goal",
            "tasks": [
                {"id": "1", "description": "task 1", "files": [], "dependencies": [], "status": "completed", "result": "done"}
            ],
        }
        state_file.write_text(json.dumps(state_data))

        client = MockLLMClient()
        config = Settings(workspace=tmpdir)

        orch = Orchestrator(
            goal="test goal",
            client=client,
            config=config,
            state_file=state_file,
        )

        assert orch.cycles == 2
        assert orch.goal == "updated goal"
        assert orch.original_goal == "original goal"
        assert len(orch.all_tasks) == 1
        assert orch.all_tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_orchestrator_save_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / ".furrow_state.json"
        client = MockLLMClient()
        config = Settings(workspace=tmpdir)

        orch = Orchestrator(
            goal="test goal",
            client=client,
            config=config,
            state_file=state_file,
        )

        orch.cycles = 1
        orch.goal = "new goal"
        orch.all_tasks = [TaskModel(id="1", description="task 1", status="completed")]

        orch._save_state()

        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["cycles"] == 1
        assert data["goal"] == "new goal"
        assert len(data["tasks"]) == 1


@pytest.mark.asyncio
async def test_orchestrator_is_done():
    client = MockLLMClient()
    config = Settings()

    orch = Orchestrator(goal="test", client=client, config=config)
    assert orch._is_done() is False  # No tasks

    orch.all_tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    assert orch._is_done() is True

    orch.all_tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="pending"),
    ]
    assert orch._is_done() is False

    orch.all_tasks = [
        TaskModel(id="1", description="task 1", status="failed"),
    ]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_max_cycles():
    with tempfile.TemporaryDirectory() as tmpdir:
        client = MockLLMClient()
        config = Settings(workspace=tmpdir, max_cycles=2)

        # Mock the agents
        with patch("furrow.core.orchestrator.PlannerAgent", MockPlannerAgent):
            with patch("furrow.core.orchestrator.WorkerAgent", MockWorkerAgent):
                with patch("furrow.core.orchestrator.TesterAgent", MockTesterAgent):
                    orch = Orchestrator(goal="test", client=client, config=config)
                    await orch.run()

        assert orch.cycles == 2


@pytest.mark.asyncio
async def test_orchestrator_cleanup_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / ".furrow_state.json"
        state_file.write_text("{}")

        client = MockLLMClient()
        config = Settings(workspace=tmpdir)

        orch = Orchestrator(goal="test", client=client, config=config, state_file=state_file)
        orch._cleanup_state()

        assert not state_file.exists()