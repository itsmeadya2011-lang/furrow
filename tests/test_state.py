from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from furrow.config import TaskModel
from furrow.core.state import (
    STATE_VERSION,
    HistoryEntry,
    RunState,
    StateStore,
    state_path_for,
)


def test_state_path_for_default_relative_path(tmp_path: Path) -> None:
    assert state_path_for(tmp_path) == tmp_path / ".furrow" / "state.json"


def test_state_path_for_custom_relative_path(tmp_path: Path) -> None:
    assert state_path_for(tmp_path, Path("custom.json")) == tmp_path / "custom.json"


def test_state_path_for_accepts_strings(tmp_path: Path) -> None:
    assert state_path_for(str(tmp_path), "custom.json") == tmp_path / "custom.json"


def test_store_starts_empty_and_saves_state(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "nested" / "dir" / "s.json")

    assert store.exists() is False
    assert store.load() is None

    store.save(RunState(goal="g"))

    assert store.exists() is True
    loaded = store.load()
    assert loaded is not None
    assert loaded.goal == "g"
    assert loaded.version == STATE_VERSION


def test_round_trip_tasks_and_history(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "s.json")
    task = TaskModel(
        id="t1",
        description="do the thing",
        files=["a.py", "b.py"],
        dependencies=["t0"],
        status="completed",
        result="all good",
    )
    entry = HistoryEntry(
        cycle=1,
        goal="ship it",
        plan_rationale="three steps",
        task_summaries=["t1 completed", "t2 pending"],
        tests_passed=True,
        test_summary="12 passed",
    )
    state = RunState(
        goal="ship it",
        original_goal="ship it",
        cycles=1,
        tasks=[task],
        history=[entry],
    )
    store.save(state)

    loaded = store.load()
    assert loaded is not None
    assert len(loaded.tasks) == 1
    assert loaded.tasks[0] == task
    assert loaded.tasks[0].id == "t1"
    assert loaded.tasks[0].description == "do the thing"
    assert loaded.tasks[0].files == ["a.py", "b.py"]
    assert loaded.tasks[0].dependencies == ["t0"]
    assert loaded.tasks[0].status == "completed"
    assert loaded.tasks[0].result == "all good"

    assert len(loaded.history) == 1
    assert loaded.history[0] == entry
    assert loaded.history[0].cycle == 1
    assert loaded.history[0].tests_passed is True
    assert loaded.history[0].task_summaries == ["t1 completed", "t2 pending"]


def test_save_leaves_no_stray_files(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store = StateStore(state_dir / "s.json")

    store.save(RunState(goal="g"))

    assert sorted(p.name for p in state_dir.iterdir()) == ["s.json"]


def test_load_returns_none_for_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text("not json {{{", encoding="utf-8")
    store = StateStore(path)

    assert store.load() is None


def test_clear_removes_file_and_is_idempotent(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "s.json")
    store.save(RunState(goal="g"))
    assert store.exists() is True

    store.clear()
    assert store.exists() is False

    store.clear()
    assert store.exists() is False


def test_save_updates_updated_at(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "s.json")
    state = RunState(goal="g")
    before = state.updated_at

    store.save(state)

    after = state.updated_at
    assert after >= before
    assert datetime.fromisoformat(after).tzinfo is not None

    loaded = store.load()
    assert loaded is not None
    assert loaded.updated_at == after
    assert datetime.fromisoformat(loaded.updated_at) >= datetime.fromisoformat(before)


def test_no_progress_and_prev_sig_roundtrip(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "s.json")
    state = RunState(goal="g", no_progress=4, prev_sig="abc|def")
    store.save(state)
    loaded = store.load()
    assert loaded is not None
    assert loaded.no_progress == 4
    assert loaded.prev_sig == "abc|def"


def test_legacy_state_without_new_keys_loads(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    legacy = {
        "version": 1,
        "goal": "g",
        "original_goal": "g",
        "cycles": 0,
        "tasks": [],
        "history": [],
        "updated_at": "2026-01-01T00:00:00+00:00",
        "usage": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")
    store = StateStore(path)
    loaded = store.load()
    assert loaded is not None
    assert loaded.no_progress == 0
    assert loaded.prev_sig == ""


def test_tests_run_round_trips_through_save_and_load(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "s.json")
    state = RunState(goal="g", tests_run=7)
    store.save(state)
    loaded = store.load()
    assert loaded is not None
    assert loaded.tests_run == 7


def test_legacy_state_without_tests_run_loads_with_the_default_zero(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    legacy = {
        "version": 1,
        "goal": "g",
        "original_goal": "g",
        "cycles": 0,
        "tasks": [],
        "history": [],
        "updated_at": "2026-01-01T00:00:00+00:00",
        "usage": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")
    store = StateStore(path)
    loaded = store.load()
    assert loaded is not None
    assert loaded.tests_run == 0

