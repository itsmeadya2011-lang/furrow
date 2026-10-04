from __future__ import annotations

from pathlib import Path

from furrow.config import (
    FileEdit,
    Plan,
    Settings,
    TaskModel,
    TestResult,
    WorkerResult,
)


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_file_edit_construction():
    e = FileEdit(path="src/foo.py", content="print('hi')\n")
    assert e.path == "src/foo.py"
    assert e.content == "print('hi')\n"


def test_file_edit_defaults():
    e = FileEdit(path="x", content="y")
    assert e.path == "x"
    assert e.content == "y"


def test_worker_result_defaults():
    r = WorkerResult(summary="done")
    assert r.summary == "done"
    assert r.edits == []


def test_worker_result_with_edits():
    r = WorkerResult(
        summary="changed",
        edits=[FileEdit(path="a.py", content="x"), FileEdit(path="b.py", content="y")],
    )
    assert len(r.edits) == 2
    assert r.edits[0].path == "a.py"


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_settings_defaults(tmp_path: Path):
    s = Settings(workspace=tmp_path)
    assert s.max_cycles == 10
    assert s.max_parallel_tasks == 5
    assert s.test_command is None
    assert s.test_timeout == 120


def test_settings_overrides(tmp_path: Path):
    s = Settings(workspace=tmp_path, max_cycles=3, max_parallel_tasks=2, test_command="pytest", test_timeout=42)
    assert s.max_cycles == 3
    assert s.max_parallel_tasks == 2
    assert s.test_command == "pytest"
    assert s.test_timeout == 42


def test_test_result_defaults():
    t = TestResult(passed=False, summary="oops")
    assert t.passed is False
    assert t.summary == "oops"
    assert t.failures == []