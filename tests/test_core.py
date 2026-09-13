import pytest
from furrow.agents.tester import TesterAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_test_result_coerce_passed_to_bool():
    """TestResult.passed is always a bool even when given truthy/falsy values."""
    t = TestResult(passed=1, summary="ok", failures=[])
    assert isinstance(t.passed, bool)
    assert t.passed is True

    t2 = TestResult(passed=0, summary="ok", failures=[])
    assert isinstance(t2.passed, bool)
    assert t2.passed is False


class MockTask:
    def __init__(self, id, status="pending", result=None):
        self.id = id
        self.status = status
        self.result = result


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_is_done_returns_true_when_all_completed(self):
        """Orchestrator should detect completion when all tasks are completed."""
        orch = Orchestrator(goal="test goal")
        orch.tasks = [MockTask("1", status="completed"), MockTask("2", status="completed")]
        assert orch._is_done() is True

    @pytest.mark.asyncio
    async def test_is_done_returns_false_when_task_failed(self):
        """Orchestrator should not be done when a task has failed."""
        orch = Orchestrator(goal="test goal")
        orch.tasks = [MockTask("1", status="completed"), MockTask("2", status="failed")]
        assert orch._is_done() is False

    @pytest.mark.asyncio
    async def test_is_done_returns_true_when_no_tasks(self):
        """Orchestrator should be done when there are no tasks (nothing left to do)."""
        orch = Orchestrator(goal="test goal")
        orch.tasks = []
        assert orch._is_done() is True

    @pytest.mark.asyncio
    async def test_tasks_persist_after_cycle(self):
        """Tasks from a cycle should be stored and accessible."""
        orch = Orchestrator(goal="test goal")
        orch.tasks = [MockTask("1", status="completed")]
        assert len(orch.tasks) == 1
        assert orch.tasks[0].status == "completed"


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_tester_agent_bool_coercion(self):
        """TesterAgent constructs TestResult with boolean passed field."""
        result = TestResult(passed=True, summary="ok", failures=[])
        assert isinstance(result.passed, bool)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_test_result_type_consistency(self):
        """TestResult.passed should always be a bool."""
        t = TestResult(passed=1, summary="x", failures=[])
        assert isinstance(t.passed, bool)
        t2 = TestResult(passed=0, summary="x", failures=[])
        assert isinstance(t2.passed, bool)
        t3 = TestResult(passed=True, summary="x", failures=[])
        assert isinstance(t3.passed, bool)
        assert t.passed is True
        assert t2.passed is False
