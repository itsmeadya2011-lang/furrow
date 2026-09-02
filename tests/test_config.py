from __future__ import annotations

from pathlib import Path

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_settings_defaults() -> None:
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.workspace == Path.cwd()


def test_task_model_defaults() -> None:
    t = TaskModel(id="t1", description="x")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_with_multiple_tasks() -> None:
    p = Plan(
        tasks=[
            TaskModel(id="t1", description="a", dependencies=["t2"]),
            TaskModel(id="t2", description="b"),
        ],
        rationale="why",
    )
    assert len(p.tasks) == 2
    assert p.tasks[0].dependencies == ["t2"]
    assert p.rationale == "why"


def test_test_result_failures_list() -> None:
    tr = TestResult(passed=False, summary="boom", failures=["err1", "err2"])
    assert tr.failures == ["err1", "err2"]


def test_provider_enum_values() -> None:
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"