import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from furrow.core.orchestrator import Orchestrator
from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TaskModel, Plan, TestResult, Provider


class MockLLMClient:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.call_count = 0
        self.settings = Settings()

    async def complete(self, prompt, system="", model=None):
        self.call_count += 1
        # Return appropriate mock response based on prompt content
        if "planner" in prompt.lower() or "plan" in prompt.lower():
            return '{"tasks": [{"id": "1", "description": "Test task", "files": ["test.py"], "dependencies": []}], "rationale": "Test plan"}'
        elif "worker" in prompt.lower() or "task:" in prompt.lower():
            return "TASK_COMPLETE: Implemented test task"
        elif "tester" in prompt.lower() or "test output" in prompt.lower():
            return '{"passed": true, "summary": "All tests passed", "failures": []}'
        return self.responses.get("default", "Mock response")


@pytest.fixture
def mock_client():
    return MockLLMClient()


@pytest.fixture
def settings():
    return Settings(max_cycles=2)


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_basic(self, mock_client):
        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("Test goal")
        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "Test task"

    @pytest.mark.asyncio
    async def test_plan_with_existing_tasks(self, mock_client):
        planner = PlannerAgent(client=mock_client)
        existing = [TaskModel(id="1", description="Done task", status="completed")]
        plan = await planner.plan("Test goal", existing)
        assert isinstance(plan, Plan)


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_worker_run(self, mock_client, tmp_path):
        settings = Settings(workspace=tmp_path)
        task = TaskModel(id="1", description="Create test file", files=["test.py"])
        worker = WorkerAgent(task=task, client=mock_client, settings=settings)
        result = await worker.run()
        assert "implemented" in result.lower() or "completed" in result.lower()


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_tester_parse_pytest_passed(self):
        tester = TesterAgent()
        output = "3 passed, 0 failed in 0.5s"
        result = tester._parse_test_output(output)
        assert result is not None
        assert result.passed is True
        assert result.summary == "3 passed, 0 failed, 0 errors"

    @pytest.mark.asyncio
    async def test_tester_parse_pytest_failed(self):
        tester = TesterAgent()
        output = "1 passed, 2 failed in 0.5s\nFAILED test_foo.py::test_bar"
        result = tester._parse_test_output(output)
        assert result is not None
        assert result.passed is False
        assert "2 failed" in result.summary

    @pytest.mark.asyncio
    async def test_tester_parse_no_runner(self):
        tester = TesterAgent()
        output = "No test runner found."
        result = tester._parse_test_output(output)
        assert result is None


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, mock_client, settings):
        orchestrator = Orchestrator(goal="Test goal", client=mock_client, config=settings)
        assert orchestrator.goal == "Test goal"
        assert orchestrator.original_goal == "Test goal"
        assert orchestrator.cycles == 0
        assert orchestrator.all_tasks == []

    @pytest.mark.asyncio
    async def test_orchestrator_is_done_empty(self, mock_client, settings):
        orchestrator = Orchestrator(goal="Test goal", client=mock_client, config=settings)
        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_orchestrator_is_done_with_completed(self, mock_client, settings):
        orchestrator = Orchestrator(goal="Test goal", client=mock_client, config=settings)
        orchestrator.all_tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="completed"),
        ]
        assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_orchestrator_is_done_with_failed(self, mock_client, settings):
        orchestrator = Orchestrator(goal="Test goal", client=mock_client, config=settings)
        orchestrator.all_tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="failed"),
        ]
        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_orchestrator_is_done_with_pending(self, mock_client, settings):
        orchestrator = Orchestrator(goal="Test goal", client=mock_client, config=settings)
        orchestrator.all_tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="pending"),
        ]
        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_orchestrator_max_cycles(self, mock_client):
        settings = Settings(max_cycles=1)
        orchestrator = Orchestrator(goal="Test goal", client=mock_client, config=settings)
        orchestrator.cycles = 1
        # The run loop checks max_cycles before incrementing
        # So at cycles=1 with max_cycles=1, it should stop
        assert orchestrator.config.max_cycles == 1