"""Tests for the Orchestrator's cycle management and completion logic."""

from __future__ import annotations

from furrow.config import TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class _FakeClient:
    """A lightweight stand-in for LLMClient that doesn't need an API key."""

    def __init__(self):
        from furrow.config import Settings

        self.settings = Settings()


def make_orchestrator(
    goal="Test goal",
    max_cycles=0,
    max_parallel_tasks=5,
    tasks=None,
    last_test_result=None,
) -> Orchestrator:
    orch = Orchestrator(
        goal=goal,
        client=_FakeClient(),
        max_cycles=max_cycles,
        max_parallel_tasks=max_parallel_tasks,
    )
    if tasks is not None:
        orch.tasks = tasks
    orch.last_test_result = last_test_result
    return orch


# ---------------------------------------------------------------------------
# _is_done
# ---------------------------------------------------------------------------
class TestIsDone:
    def test_no_tasks_means_done(self):
        orch = make_orchestrator()
        orch.tasks = []
        assert orch._is_done() is True

    def test_all_tasks_completed_and_tests_passed(self):
        tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="completed"),
        ]
        orch = make_orchestrator(
            tasks=tasks,
            last_test_result=TestResult(passed=True, summary="ok"),
        )
        assert orch._is_done() is True

    def test_all_tasks_completed_but_tests_failed(self):
        tasks = [TaskModel(id="1", description="task 1", status="completed")]
        orch = make_orchestrator(
            tasks=tasks,
            last_test_result=TestResult(passed=False, summary="fail", failures=["x"]),
        )
        assert orch._is_done() is False

    def test_failed_task_keeps_running(self):
        tasks = [TaskModel(id="1", description="task 1", status="failed")]
        orch = make_orchestrator(tasks=tasks)
        assert orch._is_done() is False

    def test_pending_task_keeps_running(self):
        tasks = [TaskModel(id="1", description="task 1", status="pending")]
        orch = make_orchestrator(tasks=tasks)
        assert orch._is_done() is False

    def test_in_progress_task_keeps_running(self):
        tasks = [TaskModel(id="1", description="task 1", status="in_progress")]
        orch = make_orchestrator(tasks=tasks)
        assert orch._is_done() is False

    def test_no_test_result_is_done_if_tasks_complete(self):
        """When last_test_result is None (no tests run), completed tasks → done."""
        tasks = [TaskModel(id="1", description="task 1", status="completed")]
        orch = make_orchestrator(tasks=tasks, last_test_result=None)
        assert orch._is_done() is True

    def test_max_cycles_zero_means_unlimited(self):
        orch = make_orchestrator(max_cycles=0)
        orch.tasks = [TaskModel(id="1", description="t", status="pending")]
        orch.cycles = 100
        assert orch._is_done() is False

    def test_max_cycles_reached(self):
        orch = make_orchestrator(max_cycles=3)
        orch.cycles = 3
        orch.tasks = []
        assert orch._is_done() is True

    def test_max_cycles_not_yet_reached(self):
        orch = make_orchestrator(max_cycles=5)
        orch.cycles = 3
        orch.tasks = [TaskModel(id="1", description="t", status="pending")]
        assert orch._is_done() is False


# ---------------------------------------------------------------------------
# _merge_tasks
# ---------------------------------------------------------------------------
class TestMergeTasks:
    def test_preserves_completed_status(self):
        old = TaskModel(id="1", description="old", status="completed", result="done")
        orch = make_orchestrator(tasks=[old])
        merged = orch._merge_tasks([TaskModel(id="1", description="old", status="pending")])
        assert merged[0].status == "completed"
        assert merged[0].result == "done"

    def test_new_tasks_are_pending(self):
        orch = make_orchestrator(tasks=[])
        merged = orch._merge_tasks([TaskModel(id="2", description="new", status="pending")])
        assert merged[0].status == "pending"

    def test_mixed_old_and_new(self):
        orch = make_orchestrator(tasks=[TaskModel(id="1", description="old", status="completed")])
        merged = orch._merge_tasks(
            [
                TaskModel(id="1", description="old", status="pending"),
                TaskModel(id="2", description="new", status="pending"),
            ]
        )
        assert len(merged) == 2
        assert merged[0].status == "completed"
        assert merged[1].status == "pending"


# ---------------------------------------------------------------------------
# _update_dependent_statuses
# ---------------------------------------------------------------------------
class TestUpdateDependentStatuses:
    def test_dependency_blocked_when_not_completed(self):
        orch = make_orchestrator(
            tasks=[
                TaskModel(id="1", description="a", status="pending"),
                TaskModel(id="2", description="b", status="pending", dependencies=["1"]),
            ]
        )
        orch._update_dependent_statuses()
        # Task 1 has no deps, stays pending; task 2 has dep on 1 which isn't
        # completed, so it stays pending (but this is already the default).
        assert orch.tasks[0].status == "pending"
        assert orch.tasks[1].status == "pending"

    def test_dependency_unblocked_when_completed(self):
        orch = make_orchestrator(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="in_progress", dependencies=["1"]),
            ]
        )
        orch._update_dependent_statuses()
        # Task 1 is completed, task 2's dependency is satisfied — but it was
        # "in_progress" not "completed", so it stays as-is.
        assert orch.tasks[0].status == "completed"
        assert orch.tasks[1].status == "in_progress"

    def test_completed_tasks_not_reset(self):
        orch = make_orchestrator(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="completed", dependencies=["1"]),
            ]
        )
        orch._update_dependent_statuses()
        assert all(t.status == "completed" for t in orch.tasks)


# ---------------------------------------------------------------------------
# _get_tasks
# ---------------------------------------------------------------------------
class TestGetTasks:
    def test_returns_self_tasks(self):
        tasks = [TaskModel(id="1", description="t")]
        orch = make_orchestrator(tasks=tasks)
        assert orch._get_tasks() is orch.tasks


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------
class TestConstructor:
    def test_defaults_from_settings(self):
        orch = make_orchestrator()
        assert orch.max_cycles == 0  # default in Settings
        assert orch.max_parallel_tasks == 5  # default in Settings
        assert orch.state_file is None  # no state file by default

    def test_state_file_from_resume(self):
        orch = Orchestrator(
            goal="test",
            client=_FakeClient(),
            max_cycles=0,
            max_parallel_tasks=5,
            resume=True,
        )
        assert orch.state_file is not None

    def test_explicit_state_file(self):
        orch = Orchestrator(
            goal="test",
            client=_FakeClient(),
            state_file="/tmp/test-furrow-state.json",
        )
        assert str(orch.state_file) == "/tmp/test-furrow-state.json"
