"""Tests for :mod:`furrow.config` (settings, models, public surface, logging).

These tests never touch the network and never read real credentials: the
``FURROW_``-prefixed environment is driven through ``monkeypatch``.  The logging
tests snapshot and restore ``structlog``'s global configuration because
:func:`configure_logging` mutates process-wide state.
"""

from __future__ import annotations

import pytest
import structlog
from pydantic import ValidationError

from furrow.config import (
    Plan,
    Provider,
    Settings,
    TaskModel,
    TestResult,
    configure_logging,
    settings,
)

# --------------------------------------------------------------------------------------
# Settings: FURROW_-prefixed environment parsing
# --------------------------------------------------------------------------------------

FURROW_VARS = (
    "FURROW_PROVIDER",
    "FURROW_MODEL",
    "FURROW_PLANNER_MODEL",
    "FURROW_WORKER_MODEL",
    "FURROW_TESTER_MODEL",
    "FURROW_ANTHROPIC_API_KEY",
    "FURROW_OPENAI_API_KEY",
    "FURROW_OLLAMA_BASE_URL",
    "FURROW_MAX_PARALLEL_TASKS",
    "FURROW_MAX_CYCLES",
    "FURROW_WORKSPACE",
    "FURROW_LOG_LEVEL",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Ensure no ``FURROW_`` variable or ``.env`` file leaks into these assertions."""
    for name in FURROW_VARS:
        monkeypatch.delenv(name, raising=False)
    # Keep the cwd out of it: ``Settings`` reads ``.env`` relative to the process cwd.
    monkeypatch.chdir("/")
    return monkeypatch


def test_defaults_without_env(clean_env: pytest.MonkeyPatch) -> None:
    loaded = Settings()

    assert loaded.provider is Provider.ANTHROPIC
    assert loaded.model == "claude-sonnet-4-20250514"
    assert loaded.planner_model == "claude-3-5-haiku-20241022"
    assert loaded.worker_model == "claude-3-5-sonnet-20241022"
    assert loaded.tester_model == "claude-3-5-sonnet-20241022"
    assert loaded.anthropic_api_key is None
    assert loaded.openai_api_key is None
    assert loaded.ollama_base_url == "http://localhost:11434"
    assert loaded.max_parallel_tasks == 5
    assert loaded.max_cycles == 0
    assert loaded.log_level == "INFO"


def test_env_prefix_furrow_is_honoured(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("FURROW_MAX_PARALLEL_TASKS", "9")
    clean_env.setenv("FURROW_MAX_CYCLES", "4")
    clean_env.setenv("FURROW_MODEL", "gpt-4o-mini")

    loaded = Settings()

    assert loaded.max_parallel_tasks == 9
    assert isinstance(loaded.max_parallel_tasks, int)
    assert loaded.max_cycles == 4
    assert loaded.model == "gpt-4o-mini"


def test_env_parses_enum_provider_and_path_workspace(
    clean_env: pytest.MonkeyPatch, tmp_path
) -> None:
    clean_env.setenv("FURROW_PROVIDER", "ollama")
    clean_env.setenv("FURROW_WORKSPACE", str(tmp_path))

    loaded = Settings()

    assert loaded.provider is Provider.OLLAMA
    assert loaded.workspace == tmp_path


def test_unprefixed_env_var_is_ignored(clean_env: pytest.MonkeyPatch) -> None:
    """Regression guard: the prefix is mandatory, not optional."""
    clean_env.setenv("MAX_PARALLEL_TASKS", "11")

    assert Settings().max_parallel_tasks == 5


def test_bad_env_value_raises_validation_error(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("FURROW_MAX_PARALLEL_TASKS", "not-a-number")

    with pytest.raises(ValidationError):
        Settings()


def test_module_level_settings_is_a_settings_instance() -> None:
    assert isinstance(settings, Settings)


# --------------------------------------------------------------------------------------
# Model defaults
# --------------------------------------------------------------------------------------


def test_task_model_defaults() -> None:
    task = TaskModel(id="1", description="do the thing")

    assert task.status == "pending"
    assert task.result is None
    assert task.files == []
    assert task.dependencies == []


def test_task_model_lists_are_not_shared() -> None:
    """``default_factory`` per instance, not a single shared mutable default."""
    first = TaskModel(id="1", description="a")
    second = TaskModel(id="2", description="b")

    first.files.append("src/a.py")
    first.dependencies.append("0")

    assert second.files == []
    assert second.dependencies == []


def test_plan_requires_tasks_and_rationale() -> None:
    plan = Plan(tasks=[{"id": "1", "description": "d"}], rationale="because")

    assert len(plan.tasks) == 1
    assert isinstance(plan.tasks[0], TaskModel)

    with pytest.raises(ValidationError):
        Plan(tasks=[])  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Plan(rationale="r")  # type: ignore[call-arg]


def test_test_result_defaults() -> None:
    result = TestResult(passed=True, summary="ok")

    assert result.failures == []
    assert result.passed is True
    assert result.summary == "ok"

    failures = TestResult(passed=False, summary="bad")
    other = TestResult(passed=False, summary="bad")
    failures.failures.append("boom")
    assert other.failures == []


# --------------------------------------------------------------------------------------
# Guards against renames / field drift / pytest collection
# --------------------------------------------------------------------------------------


def test_test_result_field_set_is_stable() -> None:
    assert sorted(TestResult.model_fields) == ["failures", "passed", "summary"]


def test_test_result_opts_out_of_pytest_collection() -> None:
    assert TestResult.__test__ is False
    assert TestResult(passed=True, summary="ok").__test__ is False


@pytest.mark.parametrize(
    ("module_name", "attr"),
    [
        ("furrow.config", "Provider"),
        ("furrow.config", "TaskModel"),
        ("furrow.config", "Plan"),
        ("furrow.config", "TestResult"),
        ("furrow.config", "Settings"),
        ("furrow.config", "settings"),
        ("furrow.config", "configure_logging"),
    ],
)
def test_public_symbols_exist(module_name: str, attr: str) -> None:
    import importlib

    module = importlib.import_module(module_name)
    assert hasattr(module, attr), f"{module_name}.{attr} disappeared"


def test_provider_enum_members() -> None:
    assert Provider.OLLAMA.value == "ollama"
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert {p.value for p in Provider} == {"anthropic", "openai", "ollama"}


# --------------------------------------------------------------------------------------
# configure_logging
# --------------------------------------------------------------------------------------


@pytest.fixture
def restore_structlog() -> object:
    """``configure_logging`` mutates global state; put it back afterwards."""
    saved = structlog.get_config()
    try:
        yield saved
    finally:
        structlog.configure(**saved)


@pytest.mark.parametrize("level", [None, "debug", "DEBUG", "nonsense-level", "", "warning"])
def test_configure_logging_does_not_raise(restore_structlog, level: str | None) -> None:
    configure_logging(level)
    assert structlog.get_config()["wrapper_class"] is not None


def test_configure_logging_unknown_level_falls_back_to_info(
    restore_structlog, capsys: pytest.CaptureFixture[str]
) -> None:
    def emit(method: str) -> str:
        logger = structlog.get_logger("furrow.tests.config.probe")
        getattr(logger, method)("probe")
        return capsys.readouterr().out

    configure_logging("definitely-not-a-level")

    assert emit("info").strip() != ""
    assert emit("debug").strip() == ""


def test_configure_logging_debug_enables_debug(
    restore_structlog, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("debug")

    logger = structlog.get_logger("furrow.tests.config.probe")
    logger.debug("probe")
    assert "probe" in capsys.readouterr().out