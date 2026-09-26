from __future__ import annotations

import json

import pytest

from furrow.config import TaskModel, TestResult
from furrow.core.state import State, StateManager


@pytest.fixture
def tmp_state_path(tmp_path):
    return tmp_path / "state.json"


@pytest.mark.asyncio
async def test_state_save_load_roundtrip(tmp_state_path):
    mgr = StateManager(tmp_state_path)
    original = State(
        goal="build a thing",
        cycle=3,
        tasks=[TaskModel(id="1", description="task one", status="completed")],
        test_result=TestResult(passed=True, summary="ok"),
    )
    await mgr.save(original)

    loaded = await mgr.load()
    assert loaded is not None
    assert loaded.goal == "build a thing"
    assert loaded.cycle == 3
    assert loaded.tasks[0].description == "task one"
    assert loaded.tasks[0].status == "completed"
    assert loaded.test_result.passed is True


@pytest.mark.asyncio
async def test_state_load_missing_returns_none(tmp_state_path):
    mgr = StateManager(tmp_state_path)
    assert await mgr.load() is None


@pytest.mark.asyncio
async def test_state_load_corrupt_returns_none(tmp_state_path):
    tmp_state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_state_path.write_text("{not valid json")
    mgr = StateManager(tmp_state_path)
    assert await mgr.load() is None


@pytest.mark.asyncio
async def test_state_save_creates_parent_dirs(tmp_path):
    nested = tmp_path / "deep" / "nested" / "state.json"
    mgr = StateManager(nested)
    await mgr.save(State(goal="x"))
    assert nested.exists()
    data = json.loads(nested.read_text())
    assert data["goal"] == "x"


@pytest.mark.asyncio
async def test_state_touch_updates_timestamp(tmp_state_path):
    mgr = StateManager(tmp_state_path)
    state = State(goal="x")
    first = state.updated_at
    await mgr.save(state)
    # save calls touch(); updated_at should be >= first
    assert state.updated_at >= first
