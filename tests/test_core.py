"""Comprehensive tests for Furrow config models, orchestrator state logic, and provider routing."""

from __future__ import annotations

from furrow.config import (
    LoopState,
    Plan,
    Provider,
    Settings,
    TaskModel,
    TestResult,
)

# ---------------------------------------------------------------------------
# Config model tests
# ---------------------------------------------------------------------------

class TestTaskModel:
    def test_default_status_is_pending(self) -> None:
        t = TaskModel(id="1", description="do thing")
        assert t.status == "pending"
        assert t.result is None

    def test_default_empty_lists(self) -> None:
        t = TaskModel(id="1", description="do thing")
        assert t.files == []
        assert t.dependencies == []

    def test_custom_fields(self) -> None:
        t = TaskModel(
            id="2",
            description="add auth",
            files=["src/auth.py"],
            dependencies=["1"],
            status="completed",
            result="done",
        )
        assert t.id == "2"
        assert t.files == ["src/auth.py"]
        assert t.dependencies == ["1"]
        assert t.status == "completed"
        assert t.result == "done"


class TestPlan:
    def test_plan_with_tasks(self) -> None:
        p = Plan(
            tasks=[TaskModel(id="1", description="task one"), TaskModel(id="2", description="task two")],
            rationale="A test plan",
        )
        assert len(p.tasks) == 2
        assert p.tasks[0].description == "task one"

    def test_plan_empty_tasks(self) -> None:
        p = Plan(tasks=[], rationale="nothing to do")
        assert p.tasks == []


class TestTestResult:
    def test_passing_result(self) -> None:
        t = TestResult(passed=True, summary="All good", failures=[])
        assert t.passed is True
        assert t.failures == []

    def test_failing_result(self) -> None:
        t = TestResult(passed=False, summary="2 failures", failures=["test_a failed", "test_b failed"])
        assert t.passed is False
        assert len(t.failures) == 2


class TestLoopState:
    def test_defaults(self) -> None:
        s = LoopState(goal="build a thing")
        assert s.goal == "build a thing"
        assert s.cycles == 0
        assert s.tasks == []
        assert s.completed is False

    def test_with_values(self) -> None:
        s = LoopState(
            goal="g",
            cycles=3,
            tasks=[TaskModel(id="1", description="t")],
            history=["cycle 1", "cycle 2"],
            completed=True,
        )
        assert s.cycles == 3
        assert len(s.tasks) == 1
        assert s.completed is True


class TestSettings:
    def test_default_provider(self) -> None:
        s = Settings()
        assert s.provider == Provider.ANTHROPIC

    def test_default_max_cycles_is_infinite(self) -> None:
        s = Settings()
        assert s.max_cycles == 0

    def test_default_ollama_url(self) -> None:
        s = Settings()
        assert s.ollama_base_url == "http://localhost:11434"

    def test_custom_provider(self) -> None:
        s = Settings(provider=Provider.OPENAI)
        assert s.provider == Provider.OPENAI

    def test_custom_max_cycles(self) -> None:
        s = Settings(max_cycles=5)
        assert s.max_cycles == 5


class TestProviderEnum:
    def test_provider_values(self) -> None:
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.OPENAI.value == "openai"
        assert Provider.OLLAMA.value == "ollama"

    def test_provider_from_string(self) -> None:
        assert Provider("ollama") == Provider.OLLAMA


# ---------------------------------------------------------------------------
# Orchestrator state logic tests (pure, no LLM calls)
# ---------------------------------------------------------------------------

class TestOrchestratorState:
    """Tests that exercise _should_terminate and _get_tasks without making LLM calls."""

    def test_no_tasks_terminates(self) -> None:
        from furrow.config import settings as default_settings
        from furrow.core.orchestrator import Orchestrator

        orch = Orchestrator.__new__(Orchestrator)
        orch.settings = default_settings
        orch.goal = "test"
        orch.tasks = []
        orch.test_results = []
        orch.cycles = 1
        assert orch._should_terminate() is True

    def test_pending_tasks_do_not_terminate(self) -> None:
        from furrow.config import settings as default_settings
        from furrow.core.orchestrator import Orchestrator

        task = TaskModel(id="1", description="do thing", status="pending")
        orch = Orchestrator.__new__(Orchestrator)
        orch.settings = default_settings
        orch.goal = "test"
        orch.tasks = [task]
        orch.test_results = []
        orch.cycles = 1
        assert orch._should_terminate() is False

    def test_failed_tasks_do_not_terminate(self) -> None:
        from furrow.config import settings as default_settings
        from furrow.core.orchestrator import Orchestrator

        task = TaskModel(id="1", description="do thing", status="failed")
        orch = Orchestrator.__new__(Orchestrator)
        orch.settings = default_settings
        orch.goal = "test"
        orch.tasks = [task]
        orch.test_results = []
        orch.cycles = 1
        assert orch._should_terminate() is False

    def test_completed_tasks_terminate(self) -> None:
        from furrow.config import settings as default_settings
        from furrow.core.orchestrator import Orchestrator

        task = TaskModel(id="1", description="do thing", status="completed")
        orch = Orchestrator.__new__(Orchestrator)
        orch.settings = default_settings
        orch.goal = "test"
        orch.tasks = [task]
        orch.test_results = [TestResult(passed=True, summary="ok", failures=[])]
        orch.cycles = 1
        assert orch._should_terminate() is True

    def test_completed_tasks_but_tests_failed_does_not_terminate(self) -> None:
        """Even if all tasks are done, a failed test should keep the loop going."""
        from furrow.config import settings as default_settings
        from furrow.core.orchestrator import Orchestrator

        task = TaskModel(id="1", description="do thing", status="completed")
        orch = Orchestrator.__new__(Orchestrator)
        orch.settings = default_settings
        orch.goal = "test"
        orch.tasks = [task]
        orch.test_results = [TestResult(passed=False, summary="fail", failures=["broken"])]
        orch.cycles = 1
        assert orch._should_terminate() is False

    def test_max_cycles_zero_is_infinite(self) -> None:
        from furrow.core.orchestrator import Orchestrator

        orch = Orchestrator.__new__(Orchestrator)
        orch.settings = Settings(max_cycles=0)
        orch.goal = "test"
        orch.tasks = [TaskModel(id="1", description="t", status="completed")]
        orch.test_results = [TestResult(passed=True, summary="ok", failures=[])]
        orch.cycles = 1
        # With max_cycles=0, completed tasks + passing tests => done
        assert orch._should_terminate() is True

    def test_max_cycles_one_terminates_after_one_cycle(self) -> None:
        from furrow.core.orchestrator import Orchestrator

        orch = Orchestrator.__new__(Orchestrator)
        orch.settings = Settings(max_cycles=1)
        orch.goal = "test"
        orch.tasks = []
        orch.test_results = []
        orch.cycles = 1
        assert orch._should_terminate() is True

    def test_max_cycles_not_yet_reached(self) -> None:
        """When max_cycles is set but not yet reached, the loop should continue."""
        from furrow.core.orchestrator import Orchestrator

        orch = Orchestrator.__new__(Orchestrator)
        orch.settings = Settings(max_cycles=3)
        orch.goal = "test"
        orch.tasks = [TaskModel(id="1", description="t", status="pending")]
        orch.test_results = []
        orch.cycles = 2
        assert orch._should_terminate() is False

    def test_get_tasks_returns_self_tasks(self) -> None:
        from furrow.config import settings as default_settings
        from furrow.core.orchestrator import Orchestrator

        orch = Orchestrator.__new__(Orchestrator)
        orch.settings = default_settings
        orch.goal = "test"
        tasks = [TaskModel(id="1", description="t")]
        orch.tasks = tasks
        orch.test_results = []
        orch.cycles = 0
        assert orch._get_tasks() is tasks

    def test_all_tasks_completed_mixed_status(self) -> None:
        from furrow.config import settings as default_settings
        from furrow.core.orchestrator import Orchestrator

        tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="pending"),
        ]
        orch = Orchestrator.__new__(Orchestrator)
        orch.settings = default_settings
        orch.goal = "test"
        orch.tasks = tasks
        orch.test_results = []
        orch.cycles = 1
        assert orch._should_terminate() is False

    def test_failed_test_keeps_loop_going_even_with_no_tasks(self) -> None:
        """If the last test failed, the loop should continue even with an empty task list."""
        from furrow.core.orchestrator import Orchestrator

        orch = Orchestrator.__new__(Orchestrator)
        orch.settings = Settings(max_cycles=0)
        orch.goal = "test"
        orch.tasks = []
        orch.test_results = [TestResult(passed=False, summary="fail", failures=["x"])]
        orch.cycles = 1
        assert orch._should_terminate() is False
