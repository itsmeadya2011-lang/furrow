from __future__ import annotations

from pathlib import Path

from furrow.core.state import clear_state, load_state, save_state


def test_save_load_round_trip(tmp_path: Path):
    p = save_state({"goal": "x"}, workspace=tmp_path)
    assert p == tmp_path / ".furrow" / "state.json"
    loaded = load_state(workspace=tmp_path)
    assert loaded is not None
    assert loaded["goal"] == "x"
    assert "updated_at" in loaded


def test_load_missing(tmp_path: Path):
    assert load_state(workspace=tmp_path) is None


def test_load_corrupt(tmp_path: Path):
    state_file = tmp_path / ".furrow" / "state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text("not json")
    assert load_state(workspace=tmp_path) is None


def test_clear_state(tmp_path: Path):
    save_state({"goal": "x"}, workspace=tmp_path)
    assert clear_state(workspace=tmp_path) is True
    assert clear_state(workspace=tmp_path) is False