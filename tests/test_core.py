import asyncio
from pathlib import Path

import pytest

from furrow.agents.planner import _extract_json, _normalize_plan, _would_cycle
from furrow.agents.worker import (
    _FILE_BLOCK_RE,
    _strip_code_blocks,
    _write_files_from_response,
)
from furrow.config import Plan, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_strips_markdown_fence():
    raw = "```json\n{\"a\": 1}\n```"
    assert _extract_json(raw) == {"a": 1}


def test_extract_json_strips_plain_fence():
    raw = "```\n{\"a\": 1}\n```"
    assert _extract_json(raw) == {"a": 1}


def test_extract_json_rejects_non_object():
    with pytest.raises(ValueError):
        _extract_json("[1, 2, 3]")


def test_extract_json_rejects_garbage():
    with pytest.raises(ValueError):
        _extract_json("not json at all")


def test_normalize_plan_assigns_ids():
    plan = Plan(
        tasks=[TaskModel(description="a"), TaskModel(description="b")],
        rationale="ok",
    )
    normalized = _normalize_plan(plan)
    assert [t.id for t in normalized.tasks] == ["1", "2"]


def test_normalize_plan_renames_duplicate_ids():
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="a"),
            TaskModel(id="1", description="b"),
            TaskModel(id="1", description="c"),
        ],
        rationale="ok",
    )
    normalized = _normalize_plan(plan)
    ids = [t.id for t in normalized.tasks]
    assert ids[0] == "1"
    assert ids[1] == "1__dup1"
    assert ids[2] == "1__dup2"
    # No two tasks share an ID.
    assert len(set(ids)) == 3


def test_normalize_plan_drops_invalid_deps():
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", dependencies=["nope"]),
            TaskModel(id="2", description="b"),
        ],
        rationale="ok",
    )
    normalized = _normalize_plan(plan)
    assert normalized.tasks[0].dependencies == []


def test_normalize_plan_drops_cyclic_deps():
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", dependencies=["2"]),
            TaskModel(id="2", description="b", dependencies=["1"]),
        ],
        rationale="ok",
    )
    normalized = _normalize_plan(plan)
    assert normalized.tasks[0].dependencies == []
    assert normalized.tasks[1].dependencies == []


def test_would_cycle_direct():
    tasks = [TaskModel(id="1", dependencies=["2"]), TaskModel(id="2", dependencies=[])]
    assert _would_cycle("2", "1", tasks, []) is True


def test_would_cycle_indirect():
    tasks = [
        TaskModel(id="1", dependencies=["2"]),
        TaskModel(id="2", dependencies=["3"]),
        TaskModel(id="3", dependencies=[]),
    ]
    # Adding dep "1" to task "3" would form 1 -> 2 -> 3 -> 1.
    assert _would_cycle("3", "1", tasks, []) is True


def test_would_cycle_no_cycle():
    tasks = [TaskModel(id="1", dependencies=[]), TaskModel(id="2", dependencies=[])]
    assert _would_cycle("1", "2", tasks, []) is False


def test_file_block_regex_matches_with_lang():
    body = "```python src/x.py\nprint('hi')\n```"
    match = _FILE_BLOCK_RE.search(body)
    assert match is not None
    assert match.group("path") == "src/x.py"
    assert match.group("body") == "print('hi')\n"


def test_file_block_regex_matches_without_lang():
    body = "```src/x.py\nprint('hi')\n```"
    match = _FILE_BLOCK_RE.search(body)
    assert match is not None
    assert match.group("path") == "src/x.py"


def test_strip_code_blocks_removes_fences():
    text = "before\n```python\nx = 1\n```\nafter"
    out = _strip_code_blocks(text)
    assert "```" not in out
    assert "before" in out and "after" in out


def test_write_files_writes_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("furrow.config.settings", Settings(workspace=tmp_path))
    response = "```python src/x.py\nprint('hi')\n```"
    written = _write_files_from_response(response, ["src/x.py"])
    assert written == ["src/x.py"]
    assert (tmp_path / "src" / "x.py").read_text(encoding="utf-8") == "print('hi')\n"


def test_write_files_rejects_disallowed():
    response = "```python src/x.py\nprint('hi')\n```"
    written = _write_files_from_response(response, ["src/y.py"])
    assert written == []


def test_write_files_rejects_path_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("furrow.config.settings", Settings(workspace=tmp_path))
    response = "```python ../evil.py\nboom\n```"
    written = _write_files_from_response(response, ["../evil.py"])
    assert written == []