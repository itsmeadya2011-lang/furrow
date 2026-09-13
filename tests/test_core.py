import pytest
from furrow.config import Plan, TaskModel, TestResult, TaskStatus, Provider


class TestTaskModel:
    def test_default_status(self):
        task = TaskModel(id="1", description="do thing")
        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.files == []
        assert task.dependencies == []

    def test_custom_status(self):
        task = TaskModel(id="1", description="do thing", status=TaskStatus.COMPLETED)
        assert task.status == TaskStatus.COMPLETED

    def test_with_files_and_deps(self):
        task = TaskModel(
            id="1", description="do thing", files=["a.py", "b.py"], dependencies=["2"]
        )
        assert task.files == ["a.py", "b.py"]
        assert task.dependencies == ["2"]


class TestPlan:
    def test_plan_parse(self):
        p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
        assert p.tasks[0].description == "do thing"
        assert p.rationale == "ok"

    def test_plan_with_tasks(self):
        tasks = [
            TaskModel(id="1", description="task 1", status=TaskStatus.COMPLETED),
            TaskModel(id="2", description="task 2", status=TaskStatus.PENDING),
        ]
        p = Plan(tasks=tasks, rationale="test plan")
        assert len(p.tasks) == 2
        assert p.tasks[0].status == TaskStatus.COMPLETED


class TestTestResult:
    def test_test_result_passed(self):
        t = TestResult(passed=True, summary="ok", failures=[])
        assert t.passed is True
        assert t.summary == "ok"
        assert t.failures == []

    def test_test_result_failed(self):
        t = TestResult(
            passed=False, summary="failures", failures=["error 1", "error 2"]
        )
        assert t.passed is False
        assert len(t.failures) == 2


class TestProvider:
    def test_provider_values(self):
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.OPENAI.value == "openai"
        assert Provider.OLLAMA.value == "ollama"


class TestOrchestratorDone:
    """Test the _is_done logic via simulation."""

    def test_no_plan_is_not_done(self):
        from furrow.core.orchestrator import Orchestrator

        orch = Orchestrator(goal="test")
        assert orch._is_done() is False

    def test_all_completed_is_done(self):
        from furrow.core.orchestrator import Orchestrator
        from furrow.config import Plan, TaskModel, TaskStatus

        tasks = [
            TaskModel(id="1", description="t1", status=TaskStatus.COMPLETED),
            TaskModel(id="2", description="t2", status=TaskStatus.COMPLETED),
        ]
        orch = Orchestrator(goal="test")
        orch.plan = Plan(tasks=tasks, rationale="test")
        assert orch._is_done() is True

    def test_failed_is_not_done(self):
        from furrow.core.orchestrator import Orchestrator
        from furrow.config import Plan, TaskModel, TaskStatus

        tasks = [
            TaskModel(id="1", description="t1", status=TaskStatus.COMPLETED),
            TaskModel(id="2", description="t2", status=TaskStatus.FAILED),
        ]
        orch = Orchestrator(goal="test")
        orch.plan = Plan(tasks=tasks, rationale="test")
        assert orch._is_done() is False

    def test_incomplete_is_not_done(self):
        from furrow.core.orchestrator import Orchestrator
        from furrow.config import Plan, TaskModel, TaskStatus

        tasks = [
            TaskModel(id="1", description="t1", status=TaskStatus.COMPLETED),
            TaskModel(id="2", description="t2", status=TaskStatus.PENDING),
        ]
        orch = Orchestrator(goal="test")
        orch.plan = Plan(tasks=tasks, rationale="test")
        assert orch._is_done() is False
