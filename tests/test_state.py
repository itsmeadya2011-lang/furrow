from __future__ import annotations

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.state import FurrowState, StateStore


def _make_state(goal: str = "x") -> FurrowState:
    return FurrowState(
        goal=goal,
        cycles=2,
        current_plan=Plan(
            tasks=[TaskModel(id="t1", description="d")],
            rationale="r",
        ),
        last_test_result=TestResult(passed=True, summary="ok", failures=[]),
        done_reason="complete",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


def test_save_and_load(tmp_path) -> None:
    store = StateStore(tmp_path / "state.json")
    state = _make_state()
    store.save(state)
    loaded = store.load()
    assert loaded is not None
    assert loaded.goal == state.goal
    assert loaded.cycles == state.cycles
    assert loaded.current_plan is not None
    assert loaded.current_plan.rationale == "r"
    assert loaded.last_test_result is not None
    assert loaded.last_test_result.passed is True
    assert loaded.done_reason == "complete"


def test_load_returns_none_when_missing(tmp_path) -> None:
    store = StateStore(tmp_path / "missing.json")
    assert store.load() is None


def test_clear_removes_file(tmp_path) -> None:
    store = StateStore(tmp_path / "state.json")
    store.save(_make_state())
    assert store.exists()
    store.clear()
    assert not store.exists()
    # clearing again is a no-op
    store.clear()


def test_atomic_write(tmp_path) -> None:
    store = StateStore(tmp_path / "state.json")
    store.save(_make_state("first"))
    assert store.exists()
    s1 = store.load()
    assert s1 is not None and s1.goal == "first"

    store.save(_make_state("second"))
    assert store.exists()
    s2 = store.load()
    assert s2 is not None and s2.goal == "second"
    # No leftover tmp file
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []