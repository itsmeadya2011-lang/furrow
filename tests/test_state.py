from furrow.core.state import State, load_state, save_state


def test_state_defaults():
    s = State(goal="do something")
    assert s.goal == "do something"
    assert s.tasks == []
    assert s.cycles == 0
    assert s.completed == 0
    assert s.failed == 0
    assert s.last_updated == 0.0
    assert s.status == "running"


async def test_state_save_load_roundtrip(tmp_path):
    state = State(goal="round trip", cycles=1, completed=2, failed=3, status="done")
    path = tmp_path / "state.json"
    await save_state(state, path)
    loaded = await load_state(path)
    assert loaded is not None
    assert loaded.goal == "round trip"
    assert loaded.cycles == 1
    assert loaded.completed == 2
    assert loaded.failed == 3
    assert loaded.status == "done"


async def test_load_state_missing_file(tmp_path):
    path = tmp_path / "missing.json"
    assert await load_state(path) is None


async def test_load_state_invalid_json(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{not valid json")
    assert await load_state(path) is None


async def test_save_state_sets_timestamp(tmp_path):
    state = State(goal="timestamp")
    path = tmp_path / "state.json"
    assert state.last_updated == 0.0
    await save_state(state, path)
    assert state.last_updated > 0.0
