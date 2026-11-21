"""Tests for the ``furrow.core`` package: public re-exports and Orchestrator helpers.

The config-model coverage that used to live here has moved to ``test_config.py``;
this module focuses on the ``furrow.core`` surface and the ``Orchestrator``
helper methods that were extracted to remove duplication in the run loop.
"""

from __future__ import annotations

from furrow.config import TaskModel
from furrow.core import Event, EventBus, EventType, Orchestrator, RunState, StateStore
from furrow.core.orchestrator import Orchestrator as OrchestratorDirect


def test_core_package_reexports_orchestrator_symbols():
    # The package __init__ must re-export the same classes as their modules.
    assert Orchestrator is OrchestratorDirect
    # Each re-exported symbol is a real class (not None or a placeholder).
    for symbol in (Event, EventBus, EventType, RunState, StateStore):
        assert isinstance(symbol, type)


def test_abort_cycle_sets_stopped_reason_and_emits_error(tmp_path):
    """A failing phase emits an ERROR event, sets the stopped reason, and
    rolls back the cycle counter."""
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=_settings(tmp_path), state_store=store)
    q = orch.event_bus.subscribe()
    orch.cycles = 1  # simulate being inside a cycle

    orch._abort_cycle("planner", ValueError("boom"))

    assert orch._stopped_reason == "planner failed: boom"
    assert orch.cycles == 0  # rolled back: the cycle did not complete
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    errors = [e for e in events if e.type == EventType.ERROR]
    assert len(errors) == 1
    assert "boom" in errors[0].message


def test_abort_cycle_accumulates_usage(tmp_path):
    """A phase that aborts after the client recorded usage still accounts for it."""
    from furrow.config import Usage

    class FakeClient:
        def __init__(self):
            self.total_usage = Usage(input_tokens=10, output_tokens=5, calls=1, cost_usd=0.5)

    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(
        goal="g", settings=_settings(tmp_path), state_store=store, client=FakeClient()
    )
    orch.cycles = 1

    orch._abort_cycle("execute", RuntimeError("nope"))

    assert orch._usage.input_tokens == 10
    assert orch._usage.output_tokens == 5
    assert orch._usage.calls == 1
    assert orch._usage.cost_usd == 0.5


def test_fail_task_marks_pending_when_retries_remain(tmp_path):
    """With retry budget left, a failed task is set back to pending and a
    retry LOG event is emitted."""
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=_settings(tmp_path), state_store=store)
    q = orch.event_bus.subscribe()
    task = TaskModel(id="t1", description="d")
    task.attempts = 1

    orch._fail_task(task, "boom", max_retries=2)

    assert task.status == "pending"
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    errors = [e for e in events if e.type == EventType.ERROR]
    assert any("boom" in e.message for e in errors)
    finished = [e for e in events if e.type == EventType.TASK_FINISHED]
    assert len(finished) == 1
    assert finished[0].data["success"] is False
    logs = [e for e in events if e.type == EventType.LOG]
    assert any("retrying" in e.message for e in logs)


def test_fail_task_marks_failed_when_budget_exhausted(tmp_path):
    """With the retry budget spent, a failed task stays failed and no retry
    LOG event is emitted."""
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=_settings(tmp_path), state_store=store)
    q = orch.event_bus.subscribe()
    task = TaskModel(id="t1", description="d")
    task.attempts = 1

    orch._fail_task(task, "boom", max_retries=0)

    assert task.status == "failed"
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    logs = [e for e in events if e.type == EventType.LOG]
    assert not any("retrying" in e.message for e in logs)
    errors = [e for e in events if e.type == EventType.ERROR]
    assert any("boom" in e.message for e in errors)
    finished = [e for e in events if e.type == EventType.TASK_FINISHED]
    assert len(finished) == 1
    assert finished[0].data["success"] is False


def test_fail_task_sets_finished_at(tmp_path):
    """A failed task gets a finished_at timestamp."""
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=_settings(tmp_path), state_store=store)
    task = TaskModel(id="t1", description="d")
    task.attempts = 1

    orch._fail_task(task, "boom", max_retries=0)

    assert task.finished_at is not None
    assert task.finished_at != ""


def _settings(tmp_path):
    from furrow.config import Settings

    return Settings(_env_file=None, workspace=tmp_path, max_cycles=1)