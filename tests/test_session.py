from __future__ import annotations

import json
from pathlib import Path

import pytest

from furrow.config import Plan, SessionState, TaskModel
from furrow.core.session import (
    SessionCorruptedError,
    SessionManager,
    SessionNotFoundError,
    generate_session_id,
)


def test_session_state_roundtrip() -> None:
    plan = Plan(
        tasks=[TaskModel(id="t1", description="do thing", status="completed")],
        rationale="ok",
    )
    state = SessionState(
        session_id="abc",
        goal="build it",
        current_goal="build it",
        cycles=3,
        current_plan=plan.model_dump(),
        status="running",
        workspace="/tmp",
    )
    raw = state.to_json()
    loaded = SessionState.from_json(raw)
    assert loaded.session_id == "abc"
    assert loaded.goal == "build it"
    assert loaded.cycles == 3
    assert loaded.current_plan == plan.model_dump()
    assert loaded.status == "running"


def test_session_state_touch_updates_timestamp() -> None:
    state = SessionState(session_id="x", goal="g", current_goal="g")
    old = state.updated_at
    state.touch()
    assert state.updated_at >= old


def test_generate_session_id_format() -> None:
    sid = generate_session_id()
    parts = sid.split("-")
    assert len(parts) == 2
    assert len(parts[0]) == 15  # YYYYMMDDTHHMMSS
    assert len(parts[1]) == 8


def test_session_manager_save_load_list_delete(tmp_path: Path) -> None:
    mgr = SessionManager(tmp_path)
    state = SessionState(
        session_id=generate_session_id(),
        goal="goal",
        current_goal="goal",
        cycles=2,
        status="running",
        workspace=str(tmp_path),
    )
    mgr.save(state.session_id, state)
    assert mgr.exists(state.session_id)

    loaded = mgr.load(state.session_id)
    assert loaded.goal == "goal"
    assert loaded.cycles == 2

    listed = mgr.list_sessions()
    assert any(s.session_id == state.session_id for s in listed)

    assert mgr.delete(state.session_id) is True
    assert not mgr.exists(state.session_id)


def test_session_manager_load_missing(tmp_path: Path) -> None:
    mgr = SessionManager(tmp_path)
    with pytest.raises(SessionNotFoundError):
        mgr.load("does-not-exist")


def test_session_manager_load_corrupted(tmp_path: Path) -> None:
    mgr = SessionManager(tmp_path)
    sid = generate_session_id()
    path = mgr.sessions_dir / f"{sid}.json"
    mgr.sessions_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(SessionCorruptedError):
        mgr.load(sid)


def test_session_manager_list_sessions_skips_corrupted(tmp_path: Path) -> None:
    mgr = SessionManager(tmp_path)
    mgr.sessions_dir.mkdir(parents=True, exist_ok=True)
    # A valid one
    good = SessionState(session_id="good", goal="g", current_goal="g")
    mgr.save("good", good)
    # A corrupted one
    (mgr.sessions_dir / "bad.json").write_text("oops", encoding="utf-8")
    sessions = mgr.list_sessions()
    ids = {s.session_id for s in sessions}
    assert "good" in ids
    assert "bad" not in ids


def test_session_manager_invalid_id_rejected(tmp_path: Path) -> None:
    mgr = SessionManager(tmp_path)
    with pytest.raises(Exception):
        mgr.load("../escape")


def test_new_session_creates_file(tmp_path: Path) -> None:
    mgr = SessionManager(tmp_path)
    sid, state = mgr.new_session(goal="hello", workspace=tmp_path)
    assert mgr.exists(sid)
    assert state.goal == "hello"
    assert state.cycles == 0
    assert state.status == "running"
