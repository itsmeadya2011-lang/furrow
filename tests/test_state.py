"""Tests for :class:`~furrow.core.state.StateStore` persistence.

The store is best-effort by design: a missing or corrupt state file must never
crash a run, writes are atomic so a crash mid-save cannot leave a half-written
file behind, and ``load()`` never raises. Those guarantees are what these tests
pin down, together with the default ``<workspace>/.furrow/state.json`` location.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import furrow.core.state as state_module
from furrow.config import TaskModel, TaskStatus
from furrow.core.state import CycleRecord, RunState, StateStore
from tests.conftest import SettingsFactory

# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #


def test_path_defaults_to_state_json_under_the_workspace(
    settings_factory: SettingsFactory,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    store = StateStore(settings_factory(workspace=workspace, state_file=None))
    assert store.path() == workspace / ".furrow" / "state.json"


def test_path_prefers_an_explicit_state_file(
    settings_factory: SettingsFactory,
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "nested" / "run.json"
    store = StateStore(settings_factory(state_file=explicit))
    assert store.path() == explicit


def test_path_follows_a_workspace_changed_after_construction(
    settings_factory: SettingsFactory,
    tmp_path: Path,
) -> None:
    settings = settings_factory(state_file=None)
    store = StateStore(settings)
    settings.workspace = tmp_path / "moved"
    assert store.path() == tmp_path / "moved" / ".furrow" / "state.json"


# --------------------------------------------------------------------------- #
# Round trips
# --------------------------------------------------------------------------- #


def test_save_then_load_round_trips_goal_cycles_tasks_and_history(
    settings_factory: SettingsFactory,
) -> None:
    store = StateStore(settings_factory())
    state = RunState(
        goal="add auth",
        cycles=2,
        tasks=[
            TaskModel(
                id="t1",
                description="write the token helper",
                files=["auth.py"],
                dependencies=["t0"],
                status=TaskStatus.COMPLETED,
                result="done",
                attempts=1,
            ),
            TaskModel(
                id="t2",
                description="write the tests",
                status=TaskStatus.FAILED,
                error="AssertionError",
                attempts=2,
            ),
        ],
        history=[
            CycleRecord(
                cycle=1,
                passed=False,
                summary="1 failed",
                failures=["tests/test_auth.py"],
                task_ids=["t1"],
                completed=["t1"],
                failed=[],
            ),
            CycleRecord(
                cycle=2,
                passed=True,
                summary="all green",
                task_ids=["t1", "t2"],
                completed=["t1"],
                failed=["t2"],
            ),
        ],
    )
    store.save(state)

    restored = store.load()

    assert restored is not None
    assert restored.goal == "add auth"
    assert restored.cycles == 2
    assert restored.tasks == state.tasks
    assert restored.history == state.history


def test_save_stamps_a_parsable_utc_timestamp(settings_factory: SettingsFactory) -> None:
    store = StateStore(settings_factory())
    state = RunState(goal="g")
    store.save(state)
    loaded = store.load()
    assert loaded is not None
    assert loaded.updated_at == state.updated_at
    parsed = datetime.fromisoformat(loaded.updated_at)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)


def test_save_creates_missing_parent_directories(settings_factory: SettingsFactory) -> None:
    target = settings_factory().workspace / "deep" / "deeper" / "state.json"
    store = StateStore(settings_factory(state_file=target))
    store.save(RunState(goal="g"))
    assert target.is_file()


def test_save_overwrites_a_previous_state(settings_factory: SettingsFactory) -> None:
    store = StateStore(settings_factory())
    store.save(RunState(goal="first", cycles=1))
    store.save(RunState(goal="second", cycles=2))
    loaded = store.load()
    assert loaded is not None
    assert loaded.goal == "second"
    assert loaded.cycles == 2


def test_save_leaves_no_temporary_file_behind(settings_factory: SettingsFactory) -> None:
    store = StateStore(settings_factory())
    store.save(RunState(goal="g"))
    assert sorted(entry.name for entry in store.path().parent.iterdir()) == ["state.json"]


def test_save_removes_the_temporary_file_when_the_rename_fails(
    settings_factory: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_replace(_source: str, _target: str) -> None:
        raise OSError("rename failed")

    monkeypatch.setattr(state_module.os, "replace", failing_replace)
    store = StateStore(settings_factory())
    store.save(RunState(goal="g"))
    assert list(store.path().parent.iterdir()) == []


def test_save_swallows_filesystem_errors(
    settings_factory: SettingsFactory,
    tmp_path: Path,
) -> None:
    # The parent path is a regular file, so mkdir() cannot succeed.
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    store = StateStore(settings_factory(state_file=blocker / "nested" / "state.json"))
    store.save(RunState(goal="g"))
    assert not (blocker / "nested").exists()


# --------------------------------------------------------------------------- #
# load() never raises
# --------------------------------------------------------------------------- #


def test_load_returns_none_when_nothing_was_saved(settings_factory: SettingsFactory) -> None:
    assert StateStore(settings_factory()).load() is None


def test_load_returns_none_for_invalid_json(settings_factory: SettingsFactory) -> None:
    store = StateStore(settings_factory())
    store.path().parent.mkdir(parents=True, exist_ok=True)
    store.path().write_text("{not json at all", encoding="utf-8")
    assert store.load() is None


def test_load_returns_none_for_json_of_the_wrong_shape(
    settings_factory: SettingsFactory,
) -> None:
    store = StateStore(settings_factory())
    store.path().parent.mkdir(parents=True, exist_ok=True)
    store.path().write_text(json.dumps({"unexpected": True}), encoding="utf-8")
    assert store.load() is None


def test_load_returns_none_when_a_task_fails_validation(
    settings_factory: SettingsFactory,
) -> None:
    store = StateStore(settings_factory())
    store.path().parent.mkdir(parents=True, exist_ok=True)
    store.path().write_text(json.dumps({"goal": "g", "tasks": [{"id": "t1"}]}), encoding="utf-8")
    assert store.load() is None


def test_load_returns_none_for_a_non_utf8_file(settings_factory: SettingsFactory) -> None:
    store = StateStore(settings_factory())
    store.path().parent.mkdir(parents=True, exist_ok=True)
    store.path().write_bytes(b"\xff\xfe\x00\x00 not utf-8")
    assert store.load() is None


def test_load_returns_none_when_the_state_path_is_a_directory(
    settings_factory: SettingsFactory,
) -> None:
    store = StateStore(settings_factory())
    store.path().mkdir(parents=True)
    assert store.load() is None


def test_load_accepts_a_state_file_with_no_optional_fields(
    settings_factory: SettingsFactory,
) -> None:
    store = StateStore(settings_factory())
    store.path().parent.mkdir(parents=True, exist_ok=True)
    store.path().write_text(json.dumps({"goal": "minimal"}), encoding="utf-8")
    loaded = store.load()
    assert loaded is not None
    assert loaded.goal == "minimal"
    assert loaded.cycles == 0
    assert loaded.tasks == []
    assert loaded.history == []


# --------------------------------------------------------------------------- #
# clear()
# --------------------------------------------------------------------------- #


def test_clear_removes_the_state_file(settings_factory: SettingsFactory) -> None:
    store = StateStore(settings_factory())
    store.save(RunState(goal="g"))
    store.clear()
    assert not store.path().exists()
    assert store.load() is None


def test_clear_is_a_no_op_when_no_state_file_exists(
    settings_factory: SettingsFactory,
) -> None:
    store = StateStore(settings_factory())
    store.clear()
    assert not store.path().exists()


def test_clear_is_a_no_op_when_the_state_path_is_a_directory(
    settings_factory: SettingsFactory,
) -> None:
    store = StateStore(settings_factory())
    store.path().mkdir(parents=True)
    store.clear()
    assert store.path().is_dir()
