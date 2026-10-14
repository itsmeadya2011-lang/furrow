from furrow.config import TaskModel, TestResult
from furrow.state import FurrowState, StateStore


class TestFurrowState:
    def test_creation_with_defaults(self):
        state = FurrowState(goal="Test goal")
        assert state.goal == "Test goal"
        assert state.cycles == 0
        assert state.tasks == []
        assert state.last_test_result is None
        assert state.status == "pending"

    def test_creation_with_values(self):
        state = FurrowState(
            goal="Test goal",
            cycles=5,
            tasks=[TaskModel(id="1", description="Task 1")],
            last_test_result=TestResult(passed=True, summary="OK"),
            status="running",
        )
        assert state.goal == "Test goal"
        assert state.cycles == 5
        assert len(state.tasks) == 1
        assert state.last_test_result.passed is True
        assert state.status == "running"


class TestStateStore:
    def test_default_path(self, tmp_path):
        store = StateStore(workspace=tmp_path)
        expected = tmp_path / ".furrow" / "state.json"
        assert store.path == expected

    def test_save_and_load_roundtrip(self, tmp_path):
        state_file = tmp_path / "state.json"
        store = StateStore(state_file)

        original = FurrowState(
            goal="Test goal",
            cycles=3,
            tasks=[TaskModel(id="1", description="Task 1")],
            last_test_result=TestResult(passed=True, summary="All good"),
        )
        store.save(original)
        loaded = store.load()

        assert loaded is not None
        assert loaded.goal == original.goal
        assert loaded.cycles == original.cycles
        assert loaded.tasks == original.tasks
        assert loaded.last_test_result == original.last_test_result

    def test_load_returns_none_for_missing_file(self, tmp_path):
        state_file = tmp_path / "nonexistent.json"
        store = StateStore(state_file)
        result = store.load()
        assert result is None

    def test_delete_removes_file(self, tmp_path):
        state_file = tmp_path / "state.json"
        store = StateStore(state_file)

        store.save(FurrowState(goal="Test"))
        assert state_file.exists()

        store.delete()
        assert not state_file.exists()

    def test_delete_on_nonexistent_file_does_not_raise(self, tmp_path):
        state_file = tmp_path / "nonexistent.json"
        store = StateStore(state_file)
        store.delete()
        assert not state_file.exists()

    def test_save_creates_parent_directories(self, tmp_path):
        nested_path = tmp_path / ".furrow" / "nested" / "state.json"
        store = StateStore(nested_path)
        store.save(FurrowState(goal="Test"))
        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_load_handles_corrupted_json(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text("not valid json")
        store = StateStore(state_file)
        result = store.load()
        assert result is None

    def test_load_handles_invalid_schema(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text('{"invalid": "schema"}')
        store = StateStore(state_file)
        result = store.load()
        assert result is None
