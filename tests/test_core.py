import pytest
from unittest.mock import AsyncMock, MagicMock

from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class MockLLMClient(LLMClient):
    def __init__(self, settings: Settings | None = None) -> None:
        if settings is None:
            settings = Settings()
        super().__init__(settings=settings)

    async def complete(self, prompt: str, system: str = "", model: str | None = None, timeout: int = 120) -> str:
        return '{"tasks": [], "rationale": "no tasks"}'

    async def read_file(self, path: str) -> str:
        return ""

    async def write_file(self, path: str, content: str) -> None:
        pass

    def list_files(self, directory: str) -> list[str]:
        return []


class MockPlanner:
    def __init__(self, plans: list[Plan]) -> None:
        self.plans = plans
        self.call_count = 0

    async def plan(self, goal: str) -> Plan:
        plan = self.plans[self.call_count % len(self.plans)]
        self.call_count += 1
        return plan


class MockTester:
    def __init__(self, result: TestResult) -> None:
        self.result = result

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        return self.result


class MockWorker:
    def __init__(self, results: list[str]) -> None:
        self.results = results
        self.call_count = 0

    async def run(self) -> str:
        result = self.results[self.call_count % len(self.results)]
        self.call_count += 1
        return result


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


@pytest.mark.asyncio
async def test_orchestrator_get_tasks_returns_plan_tasks():
    client = MockLLMClient()
    plan = Plan(tasks=[TaskModel(id="1", description="task 1")], rationale="test")
    orchestrator = Orchestrator(goal="test", client=client)

    # Before cycle, plan is None
    assert orchestrator._get_tasks() == []

    # After setting plan, tasks are returned
    orchestrator.plan = plan
    assert orchestrator._get_tasks() == plan.tasks


@pytest.mark.asyncio
async def test_orchestrator_is_done_when_all_completed():
    client = MockLLMClient()
    plan = Plan(tasks=[TaskModel(id="1", description="task 1")], rationale="test")
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator.plan = plan

    # Mark task as completed
    plan.tasks[0].status = "completed"

    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_is_done_when_has_failures():
    client = MockLLMClient()
    plan = Plan(tasks=[TaskModel(id="1", description="task 1")], rationale="test")
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator.plan = plan

    # Mark task as failed
    plan.tasks[0].status = "failed"

    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_done_respects_max_cycles():
    client = MockLLMClient()
    plan = Plan(tasks=[TaskModel(id="1", description="task 1")], rationale="test")
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator.plan = plan

    # Set max_cycles to 2
    client.settings.max_cycles = 2
    orchestrator.cycles = 2

    assert orchestrator._is_done() is True

    # Under max cycles, should not be done (even if no plan)
    orchestrator.cycles = 1
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_done_max_cycles_zero_is_unlimited():
    client = MockLLMClient()
    plan = Plan(tasks=[TaskModel(id="1", description="task 1")], rationale="test")
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator.plan = plan

    # max_cycles=0 means unlimited
    client.settings.max_cycles = 0
    orchestrator.cycles = 999

    # Should not be done just because cycles is high when max_cycles is 0
    # (it would still be done because completed >= len(tasks) though)
    plan.tasks[0].status = "pending"
    assert orchestrator._is_done() is False
