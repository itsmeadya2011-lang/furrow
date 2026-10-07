from __future__ import annotations

from furrow.config import Provider, Settings, TaskModel


def test_default_models():
    s = Settings(_env_file=None)
    assert s.model == "claude-sonnet-4-20250514"
    assert s.planner_model == "claude-haiku-4-5-20251001"
    assert s.worker_model == "claude-sonnet-4-20250514"
    assert s.tester_model == "claude-sonnet-4-20250514"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0


def test_provider_enum():
    assert Provider("ollama") == Provider.OLLAMA
    assert Provider.ANTHROPIC == "anthropic"


def test_task_model_defaults():
    t = TaskModel(id="1", description="d")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None