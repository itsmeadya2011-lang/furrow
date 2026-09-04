from __future__ import annotations

import json
from pathlib import Path

import pytest

from furrow.config import Plan, TaskModel
from furrow.core.state import StateStore


class TestStateStore:
    def test_append_and_load_latest(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.jsonl")
        store.append({"type": "plan", "cycle": 1})
        store.append({"type": "cycle_result", "cycle": 1, "passed": True})

        latest = store.load_latest()
        assert latest == {"type": "cycle_result", "cycle": 1, "passed": True}

    def test_load_latest_returns_none_when_empty(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "missing.jsonl")
        assert store.load_latest() is None

    def test_load_latest_skips_malformed_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "state.jsonl"
        p.write_text("not json\n" + json.dumps({"type": "ok", "v": 1}) + "\n")
        store = StateStore(p)
        assert store.load_latest() == {"type": "ok", "v": 1}

    def test_save_plan_serializes_tasks(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.jsonl")
        plan = Plan(
            tasks=[
                TaskModel(id="1", description="x", files=["a.py"]),
                TaskModel(id="2", description="y"),
            ],
            rationale="because",
        )
        store.save_plan("build x", cycle=3, plan=plan)
        latest = store.load_latest()
        assert latest is not None
        assert latest["type"] == "plan"
        assert latest["cycle"] == 3
        assert latest["goal"] == "build x"
        assert len(latest["tasks"]) == 2
        assert latest["tasks"][0]["files"] == ["a.py"]

    def test_save_cycle_result(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.jsonl")
        store.save_cycle_result("build x", cycle=2, passed=False, summary="bad")
        latest = store.load_latest()
        assert latest == {
            "type": "cycle_result",
            "goal": "build x",
            "cycle": 2,
            "passed": False,
            "summary": "bad",
        }

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "a" / "b" / "state.jsonl")
        store.append({"v": 1})
        assert (tmp_path / "a" / "b" / "state.jsonl").exists()