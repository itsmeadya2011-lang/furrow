from furrow.config import FurrowState, Plan, Settings, TaskModel


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("FURROW_MAX_CYCLES", "5")
    monkeypatch.setenv("FURROW_LOG_LEVEL", "DEBUG")
    s = Settings()
    assert s.max_cycles == 5
    assert s.log_level == "DEBUG"


def test_furrow_state_with_tasks(tmp_path):
    tasks = [
        TaskModel(id="1", description="first", status="completed"),
        TaskModel(id="2", description="second", status="failed"),
    ]
    state = FurrowState(
        goal="test",
        cycle=2,
        tasks=tasks,
        completed=["1"],
        failed=["2"],
    )
    path = tmp_path / "state.json"
    state.save(path)
    loaded = FurrowState.load(path)
    assert len(loaded.tasks) == 2
    assert loaded.tasks[0].status == "completed"
    assert loaded.tasks[1].status == "failed"
    assert loaded.completed == ["1"]
    assert loaded.failed == ["2"]
