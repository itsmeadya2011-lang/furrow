"""Contracts for :mod:`furrow.config`.

Status enum, task defaults, settings, logging, import safety.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow import config
from furrow.config import (
    Plan,
    Provider,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
    configure_logging,
)

# --------------------------------------------------------------------------
# TaskStatus / TaskModel
# --------------------------------------------------------------------------


def test_task_status_has_exactly_the_documented_members() -> None:
    assert {member.value for member in TaskStatus} == {
        "pending",
        "running",
        "completed",
        "failed",
        "skipped",
    }


def test_task_status_is_a_str_enum() -> None:
    assert issubclass(TaskStatus, str)
    assert TaskStatus.PENDING == "pending"
    assert isinstance(TaskStatus.PENDING, str)


def test_fresh_task_is_pending_by_both_measure() -> None:
    task = TaskModel(id="1", description="do the thing")
    assert task.status == "pending"
    assert task.status is TaskStatus.PENDING


def test_assigning_a_plain_status_string_keeps_string_comparisons_true() -> None:
    task = TaskModel(id="1", description="do the thing")
    task.status = "failed"
    assert task.status == "failed"
    assert task.status != "completed"


def test_task_mutable_defaults_are_not_shared_between_instances() -> None:
    first = TaskModel(id="1", description="one")
    second = TaskModel(id="2", description="two")

    first.files.append("a.py")
    first.dependencies.append("1")
    first.result = "x"

    assert second.files == []
    assert second.dependencies == []
    assert second.result is None


def test_task_requires_id_and_description() -> None:
    with pytest.raises(ValidationError):
        TaskModel(id="1")  # type: ignore[call-arg]


def test_plan_and_test_result_round_trip() -> None:
    plan = Plan(tasks=[TaskModel(id="1", description="ship it")], rationale="small slice")
    assert plan.tasks[0].status == "pending"
    assert plan.tasks[0].files == []

    result = TestResult(passed=True, summary="ok", failures=[])
    assert result.passed is True
    assert result.failures == []


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

#: The complete documented field set, with the default stated in furrow/config.py.
DOCUMENTED_DEFAULTS: dict[str, object] = {
    "provider": Provider.ANTHROPIC,
    "model": "claude-sonnet-4-20250514",
    "planner_model": "claude-3-5-haiku-20241022",
    "worker_model": "claude-3-5-sonnet-20241022",
    "tester_model": "claude-3-5-sonnet-20241022",
    "anthropic_api_key": None,
    "openai_api_key": None,
    "ollama_base_url": "http://localhost:11434",
    "max_parallel_tasks": 5,
    "max_cycles": 0,
    "fail_fast_cycles": 3,
    "max_tool_turns": 10,
    "test_timeout": 120,
    "test_command": None,
    "workspace": Path.cwd(),
    "log_level": "INFO",
}


def test_settings_declares_exactly_the_documented_fields() -> None:
    assert set(Settings.model_fields) == set(DOCUMENTED_DEFAULTS)


@pytest.mark.parametrize(("field", "expected"), sorted(DOCUMENTED_DEFAULTS.items()))
def test_settings_default_value(field: str, expected: object) -> None:
    settings = Settings(_env_file=None)
    assert getattr(settings, field) == expected


def test_settings_reads_furrow_prefixed_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "2")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "7")
    monkeypatch.setenv("FURROW_TEST_COMMAND", "pytest -q")
    settings = Settings(_env_file=None)
    assert settings.max_parallel_tasks == 2
    assert settings.max_cycles == 7
    assert settings.test_command == "pytest -q"


def test_settings_rejects_an_unknown_provider_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_PROVIDER", "definitely-not-a-provider")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------


def test_configure_logging_is_idempotent() -> None:
    root = logging.getLogger()
    configure_logging("DEBUG")
    after_first = len(root.handlers)
    configure_logging("DEBUG")
    configure_logging("ERROR")
    assert len(root.handlers) == after_first


def test_configure_logging_does_not_duplicate_handlers_in_a_bare_logger() -> None:
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers.clear()
    try:
        configure_logging("INFO")
        first = len(root.handlers)
        configure_logging("INFO")
        assert first >= 1
        assert len(root.handlers) == first
    finally:
        root.handlers[:] = saved


# --------------------------------------------------------------------------
# Import resilience
# --------------------------------------------------------------------------


def test_importing_config_survives_a_malformed_env_file(tmp_path: Path) -> None:
    """A bad ``.env`` must degrade to defaults, never raise a ValidationError at import."""
    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / ".env").write_text(
        "FURROW_PROVIDER=definitely-not-a-provider\nFURROW_MAX_PARALLEL_TASKS=not-a-number\n",
        encoding="utf-8",
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("FURROW_")
        and key not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}
    }
    env["PYTHONPATH"] = str(repo_root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    proc = subprocess.run(
        [sys.executable, "-c", "import furrow.config as c; print(c.settings.provider.value)"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert proc.stdout.strip().endswith("anthropic")


def test_module_level_settings_singleton_exists() -> None:
    assert isinstance(config.settings, Settings)
