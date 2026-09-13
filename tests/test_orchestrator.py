from unittest.mock import AsyncMock

import pytest

from furrow.core.orchestrator import Orchestrator
from furrow.config import TaskModel, settings


class MockLLMClient:
    def __init__(self, response="", workspace=None):
        self._response = response
        self.written_files = {}
        self.read_files = []
        self.listed_dirs = []
        self.workspace = workspace or "/tmp/furrow_test"

    @property
    def settings(self):
        class _Settings:
            workspace = "/tmp/furrow_test"
            planner_model = "test-planner"
            worker_model = "test-worker"
            tester_model = "test-tester"
        return _Settings()

    async def complete(self, prompt, system=None, model=None):
        return self._response

    async def read_file(self, path):
        self.read_files.append(str(path))
        return "mock content"

    async def write_file(self, path, content):
        self.written_files[str(path)] = content

    def list_files(self, directory):
        self.listed_dirs.append(str(directory))
        return []


@pytest.fixture
def mock_client():
    return MockLLMClient()


class TestOrchestratorIsDone:
    def test_empty_tasks_list_is_done(self, mock_client):
        orb = Orchestrator(goal="test", client=mock_client)
        orb.tasks = []
        assert orb._is_done() is True

    def test_all_completed_is_done(self, mock_client):
        orb = Orchestrator(goal="test", client=mock_client)
        orb.tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ]
        assert orb._is_done() is True

    def test_some_failed_is_not_done(self, mock_client):
        orb = Orchestrator(goal="test", client=mock_client)
        orb.tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="failed"),
        ]
        assert orb._is_done() is False

    def test_mix_pending_completed_is_not_done(self, mock_client):
        orb = Orchestrator(goal="test", client=mock_client)
        orb.tasks = [
            TaskModel(id="1", description="a", status="pending"),
            TaskModel(id="2", description="b", status="completed"),
        ]
        assert orb._is_done() is False


class TestOrchestratorGetTasks:
    def test_get_tasks_returns_persisted_tasks(self, mock_client):
        orb = Orchestrator(goal="test", client=mock_client)
        tasks = [TaskModel(id="1", description="task1", status="pending")]
        orb.tasks = tasks
        assert orb._get_tasks() is tasks
        assert orb._get_tasks() == tasks


class TestOrchestratorMaxCycles:
    def test_max_cycles_stored(self, mock_client):
        orb = Orchestrator(goal="test", client=mock_client, max_cycles=2)
        assert orb.max_cycles == 2

    def test_max_cycles_default_from_settings(self, mock_client):
        orb = Orchestrator(goal="test", client=mock_client)
        assert orb.max_cycles == settings.max_cycles


class TestOrchestratorMaxParallelTasks:
    def test_max_parallel_tasks_stored(self, mock_client):
        orb = Orchestrator(goal="test", client=mock_client, max_parallel_tasks=3)
        assert orb.max_parallel_tasks == 3

    def test_max_parallel_tasks_default_from_settings(self, mock_client):
        orb = Orchestrator(goal="test", client=mock_client)
        assert orb.max_parallel_tasks == settings.max_parallel_tasks


class TestOrchestratorEmit:
    @pytest.mark.asyncio
    async def test_emit_calls_output_callback(self, mock_client):
        callback = AsyncMock()
        orb = Orchestrator(goal="test", client=mock_client, output_callback=callback)
        await orb._emit("hello")
        callback.assert_awaited_once_with("hello")

    @pytest.mark.asyncio
    async def test_emit_without_callback(self, mock_client):
        orb = Orchestrator(goal="test", client=mock_client)
        await orb._emit("hello")
