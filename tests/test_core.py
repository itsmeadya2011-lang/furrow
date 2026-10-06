from __future__ import annotations

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from furrow.config import Plan, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="test")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_task_model_with_files():
    t = TaskModel(id="1", description="test", files=["a.py", "b.py"])
    assert t.files == ["a.py", "b.py"]


def test_settings_defaults():
    s = Settings()
    assert s.provider.value == "anthropic"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])