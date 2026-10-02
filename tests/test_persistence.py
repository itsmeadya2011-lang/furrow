from furrow.config import TaskModel, TestResult
from furrow.core.state import STATE_FILENAME, StateStore, state_from_dict


def test_default_filename_constant():
    assert STATE_FILENAME == "furrow_state.json"


def test_state_store_save_load_roundtrip(tmp_path):
    store = StateStore(tmp_path / "furrow_state.json")
    data = {
        "goal": "add JWT auth",
        "cycles": 5,
        "tasks": [{"id": "1", "description": "do thing", "status": "completed"}],
        "last_test_result": {"passed": True, "summary": "all good", "failures": []},
        "done": True,
        "updated_at": "2026-10-02T10:00:00+00:00",
    }
    store.save(data)
    loaded = store.load()
    assert loaded["goal"] == "add JWT auth"
    assert loaded["cycles"] == 5
    assert loaded["done"] is True
    assert loaded["tasks"][0]["id"] == "1"


def test_state_store_missing_file_returns_empty(tmp_path):
    store = StateStore(tmp_path / "does_not_exist.json")
    assert store.load() == {}


def test_state_store_handles_corrupt_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    store = StateStore(p)
    assert store.load() == {}


def test_state_store_clear(tmp_path):
    p = tmp_path / "state.json"
    store = StateStore(p)
    store.save({"goal": "x"})
    assert p.exists()
    store.clear()
    assert not p.exists()
    store.clear()  # idempotent: no error when file is absent


def test_state_from_dict_normalizes_types():
    data = {
        "goal": "g",
        "cycles": 1,
        "tasks": [{"id": "1", "description": "d", "status": "completed"}],
        "last_test_result": {"passed": True, "summary": "s", "failures": []},
        "done": False,
        "updated_at": None,
    }
    norm = state_from_dict(data)
    assert norm["goal"] == "g"
    assert norm["cycles"] == 1
    assert norm["done"] is False
    assert isinstance(norm["tasks"][0], TaskModel)
    assert norm["tasks"][0].id == "1"
    assert isinstance(norm["last_test_result"], TestResult)
    assert norm["last_test_result"].passed is True


def test_state_from_dict_handles_empty():
    norm = state_from_dict({})
    assert norm["goal"] == ""
    assert norm["cycles"] == 0
    assert norm["tasks"] == []
    assert norm["last_test_result"] is None
    assert norm["done"] is False
