import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class MockLLMClient:
    """A mock LLM client for testing."""

    def __init__(self, responses=None):
        self.settings = Settings()
        self.responses = responses or []
        self.call_count = 0
        self.written_files = {}

    async def complete(self, prompt, system="", model=None):
        if self.responses:
            return self.responses.pop(0)
        return "{}"

    async def read_file(self, path):
        return ""

    async def write_file(self, path, content):
        self.written_files[str(path)] = content

    def list_files(self, directory):
        return []


def test_orchestrator_init():
    client = MockLLMClient()
    orch = Orchestrator(goal="test goal", client=client)
    assert orch.goal == "test goal"
    assert orch.client == client
    assert orch.cycles == 0
    assert orch.all_tasks == {}
    assert orch.consecutive_failures == 0


def test_orchestrator_is_done_empty():
    client = MockLLMClient()
    orch = Orchestrator(goal="test", client=client)
    assert orch._is_done() is False


def test_orchestrator_is_done_all_completed():
    client = MockLLMClient()
    orch = Orchestrator(goal="test", client=client)
    orch.all_tasks = {
        "1": TaskModel(id="1", description="task1", status="completed"),
        "2": TaskModel(id="2", description="task2", status="completed"),
    }
    orch.last_test_passed = True
    assert orch._is_done() is True


def test_orchestrator_is_done_all_completed_but_tests_failed():
    client = MockLLMClient()
    orch = Orchestrator(goal="test", client=client)
    orch.all_tasks = {
        "1": TaskModel(id="1", description="task1", status="completed"),
        "2": TaskModel(id="2", description="task2", status="completed"),
    }
    orch.last_test_passed = False
    assert orch._is_done() is False


def test_orchestrator_is_done_with_failures():
    client = MockLLMClient()
    orch = Orchestrator(goal="test", client=client)
    orch.all_tasks = {
        "1": TaskModel(id="1", description="task1", status="completed"),
        "2": TaskModel(id="2", description="task2", status="failed"),
    }
    assert orch._is_done() is False


async def test_orchestrator_max_cycles():
    client = MockLLMClient()
    orch = Orchestrator(goal="test", client=client)
    orch.planner = AsyncMock()
    orch.planner.plan = AsyncMock(
        return_value=Plan(
            tasks=[TaskModel(id="1", description="do thing")],
            rationale="test",
        )
    )

    tester_mock = AsyncMock()
    tester_mock.run = AsyncMock(
        return_value=TestResult(passed=True, summary="ok", failures=[])
    )

    with patch("furrow.core.orchestrator.TesterAgent", return_value=tester_mock):
        client.settings.max_cycles = 2
        await orch.run()
        assert orch.cycles == 2


async def test_orchestrator_max_consecutive_failures():
    client = MockLLMClient()
    orch = Orchestrator(goal="test", client=client)
    orch.planner = AsyncMock()
    orch.planner.plan = AsyncMock(
        return_value=Plan(
            tasks=[TaskModel(id="1", description="do thing")],
            rationale="test",
        )
    )

    tester_mock = AsyncMock()
    tester_mock.run = AsyncMock(
        return_value=TestResult(passed=False, summary="fail", failures=["error"])
    )

    with patch("furrow.core.orchestrator.TesterAgent", return_value=tester_mock):
        client.settings.max_cycle_failures = 3
        await orch.run()
        assert orch.consecutive_failures == 3
        assert orch.cycles == 3


def test_orchestrator_get_tasks():
    client = MockLLMClient()
    orch = Orchestrator(goal="test", client=client)
    assert orch._get_tasks() == []

    task = TaskModel(id="1", description="task1")
    orch.all_tasks["1"] = task
    assert orch._get_tasks() == [task]


def test_orchestrator_console_param():
    client = MockLLMClient()
    custom_console = MagicMock()
    orch = Orchestrator(goal="test", client=client, console=custom_console)
    assert orch.console is custom_console