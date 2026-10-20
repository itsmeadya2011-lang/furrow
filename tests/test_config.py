from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import tomllib

from furrow.config import Plan, Settings, TaskModel, TestResult


def _find_pyproject(start: Path) -> Path:
    current = start.resolve()
    for parent in [current, *current.parents]:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("pyproject.toml not found")


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.provider.value == "anthropic"
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0
    assert settings.ollama_model == "llama3"


def test_settings_reads_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "3")
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    settings = Settings()
    assert settings.max_parallel_tasks == 3
    assert settings.provider.value == "openai"


def test_task_model_defaults() -> None:
    task = TaskModel(id="1", description="do thing")
    assert task.status == "pending"
    assert task.files == []
    assert task.dependencies == []
    assert task.result is None


def test_plan_and_test_result_round_trip() -> None:
    plan = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="because",
    )
    assert plan.tasks[0].id == "1"
    assert plan.rationale == "because"

    result = TestResult(passed=True, summary="ok", failures=[])
    assert result.passed is True
    assert result.summary == "ok"
    assert result.failures == []


def test_test_result_is_not_collected_by_pytest() -> None:
    assert TestResult.__test__ is False


def test_pyproject_build_backend_is_importable() -> None:
    pyproject = _find_pyproject(Path(__file__).resolve().parent)
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    assert importlib.util.find_spec("setuptools.build_meta") is not None