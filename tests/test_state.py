"""Tests for run state persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from furrow.config import Settings, TaskModel, TaskStatus
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import (
    RunState,
    load_state,
    restore_orchestrator,
    save_state,
    state_from_orchestrator,
)
from tests.helpers import FakeLLMClient


def make_orchestrator(settings: Settings) -> Orchestrator:
    return Orchestrator("original goal", client=FakeLLMClient(settings))


def test_state_round_trip(settings: Settings, tmp_path: Path) -> None:
    orch = make_orchestrator(settings)
    orch.goal = "narrowed goal"
    orch.original_goal = "original goal"
    orch.cycles = 3
    orch.tasks = [
        TaskModel(
            id="1",
            description="do it",
            status=TaskStatus.COMPLETED,
            files_changed=["a.py"],
        )
    ]
    orch.history = [
        {
            "cycle": 1,
            "tasks": ["1"],
            "passed": False,
            "summary": "boom",
            "completed": 1,
            "failed": 0,
            "failures": [],
            "satisfied": False,
        }
    ]

    state = state_from_orchestrator(orch)
    path = tmp_path / "nested" / "state.json"
    saved = save_state(state, path)

    assert saved == path
    assert path.is_file()
    loaded = load_state(path)
    assert loaded.goal == "narrowed goal"
    assert loaded.original_goal == "original goal"
    assert loaded.cycles == 3
    assert loaded.version == 1
    assert loaded.updated_at
    assert [task.id for task in loaded.tasks] == ["1"]
    assert loaded.tasks[0].status == TaskStatus.COMPLETED
    assert loaded.tasks[0].files_changed == ["a.py"]
    assert loaded.history == orch.history


def test_save_state_leaves_no_temporary_files_behind(settings: Settings, tmp_path: Path) -> None:
    orch = make_orchestrator(settings)
    path = tmp_path / "state.json"

    save_state(state_from_orchestrator(orch), path)

    entries = [entry.name for entry in tmp_path.iterdir()]
    assert "state.json" in entries
    assert not any(name.startswith(".state_") or name.endswith(".tmp") for name in entries)
    # only state.json as a file; tmp_workspace fixture may create workspace dir
    files = [e.name for e in tmp_path.iterdir() if e.is_file()]
    assert files == ["state.json"]


def test_restore_orchestrator_rehydrates_the_run(settings: Settings) -> None:
    orch = make_orchestrator(settings)
    state = RunState(
        goal="narrowed",
        original_goal="original",
        cycles=2,
        tasks=[TaskModel(id="1", description="x", status=TaskStatus.FAILED)],
        history=[{"cycle": 1, "passed": False}],
    )

    restore_orchestrator(orch, state)

    assert orch.goal == "narrowed"
    assert orch.original_goal == "original"
    assert orch.cycles == 2
    assert [task.id for task in orch.tasks] == ["1"]
    assert orch.tasks[0].status == TaskStatus.FAILED
    assert orch.history == [{"cycle": 1, "passed": False}]


def test_load_state_reports_a_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"

    with pytest.raises(FileNotFoundError, match="No saved state found"):
        load_state(missing)


def test_load_state_reports_a_corrupt_file(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="is corrupt"):
        load_state(corrupt)


def test_load_state_reports_an_invalid_payload(tmp_path: Path) -> None:
    invalid = tmp_path / "bad.json"
    invalid.write_text('{"goal": 12}', encoding="utf-8")

    with pytest.raises(ValueError, match="is invalid"):
        load_state(invalid)
