"""Tests for :mod:`furrow.config`: domain models, settings and log levels.

The log-level tests touch the process-wide root logger, so the autouse
:func:`_isolate_root_logging` fixture snapshots and restores its ``level`` and
``handlers``; without it every later test's logging behaviour would depend on
which of these ran last.

The ``get_settings()`` singleton is isolated by the autouse
``tests.conftest._isolate_settings`` fixture, which calls
``config_module.get_settings.cache_clear()`` on entry and exit. The tests that
poison the environment therefore pair :func:`monkeypatch.setenv` or
:func:`monkeypatch.chdir` with an explicit ``cache_clear()``, and the fixture's
exit-time clear means a poisoned singleton can never reach another module.
"""

from __future__ import annotations

import logging
import runpy
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

import furrow.config as config_module
from furrow.config import (
    LOG_LEVELS,
    ConfigurationError,
    Plan,
    Provider,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
    configure_logging,
    normalize_log_level,
)

#: The shipped example environment file, resolved relative to this module so the
#: test does not depend on the working directory.
ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


def _documented_env_examples() -> dict[str, str]:
    """Return every ``# FURROW_X=value`` line in ``.env.example``.

    Returns:
        A mapping of environment variable name to the example value shown on the
        right-hand side of the ``=``, which may be empty.
    """
    examples: dict[str, str] = {}
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("# FURROW_"):
            continue
        key, separator, value = line[1:].strip().partition("=")
        examples[key] = value if separator else ""
    return examples


def test_task_status_accepts_every_documented_value() -> None:
    assert {status.value for status in TaskStatus} == {
        "pending",
        "running",
        "completed",
        "failed",
        "skipped",
    }


def test_task_status_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        TaskStatus("garbage")


def test_task_status_is_a_string_enum() -> None:
    assert TaskStatus.COMPLETED == "completed"
    assert isinstance(TaskStatus.PENDING, str)


def test_task_model_defaults_are_empty_and_pending() -> None:
    task = TaskModel(id="1", description="do the thing")

    assert task.files == []
    assert task.dependencies == []
    assert task.status is TaskStatus.PENDING
    assert task.result is None


def test_task_model_accepts_an_explicit_optional_result() -> None:
    assert TaskModel(id="1", description="d", result="done").result == "done"
    assert TaskModel(id="1", description="d", result=None).result is None


def test_task_model_coerces_status_strings_and_rejects_nonsense() -> None:
    assert TaskModel(id="1", description="d", status="failed").status is TaskStatus.FAILED

    with pytest.raises(ValidationError):
        TaskModel(id="1", description="d", status="not-a-status")


def test_task_model_requires_id_and_description() -> None:
    with pytest.raises(ValidationError):
        TaskModel(id="1")
    with pytest.raises(ValidationError):
        TaskModel(description="no id")


def test_plan_defaults_to_no_tasks_and_no_rationale() -> None:
    plan = Plan()

    assert plan.tasks == []
    assert plan.rationale == ""


def test_plan_mutable_defaults_are_not_shared() -> None:
    first = Plan()
    second = Plan()

    first.tasks.append(TaskModel(id="1", description="d"))
    first.rationale = "because"

    assert second.tasks == []
    assert second.rationale == ""


def test_test_result_defaults() -> None:
    result = TestResult(passed=False)

    assert result.passed is False
    assert result.summary == ""
    assert result.failures == []


def test_test_result_is_hidden_from_pytest_collection() -> None:
    assert TestResult.__test__ is False


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.provider is Provider.ANTHROPIC
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0
    assert settings.max_tokens == 4096
    assert settings.request_timeout == 120.0
    assert settings.max_retries == 3
    assert settings.test_timeout == 120.0
    assert settings.test_command is None
    assert settings.log_level == "INFO"
    assert settings.override_model is None


def test_settings_reads_furrow_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "9")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "4")
    monkeypatch.setenv("FURROW_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("FURROW_TEST_COMMAND", "make check")

    settings = Settings(_env_file=None)

    assert settings.max_parallel_tasks == 9
    assert settings.max_cycles == 4
    assert settings.log_level == "DEBUG"
    assert settings.test_command == "make check"


def test_settings_explicit_values_beat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "9")

    settings = Settings(_env_file=None, max_parallel_tasks=2)

    assert settings.max_parallel_tasks == 2


def test_api_key_for_prefers_the_configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    settings = Settings(_env_file=None, anthropic_api_key="from-settings")

    assert settings.api_key_for(Provider.ANTHROPIC) == "from-settings"


def test_api_key_for_falls_back_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-from-env")
    settings = Settings(_env_file=None)

    assert settings.api_key_for(Provider.ANTHROPIC) == "from-env"
    assert settings.api_key_for(Provider.OPENAI) == "openai-from-env"


def test_api_key_for_is_none_when_nothing_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(_env_file=None)

    assert settings.api_key_for(Provider.ANTHROPIC) is None


def test_api_key_for_ollama_is_always_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "irrelevant")
    settings = Settings(_env_file=None, provider=Provider.OLLAMA)

    assert settings.api_key_for() is None
    assert settings.api_key_for(Provider.OLLAMA) is None


def test_api_key_for_defaults_to_the_configured_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    settings = Settings(_env_file=None, provider=Provider.OPENAI)

    assert settings.api_key_for() == settings.api_key_for(Provider.OPENAI) == "openai-key"


def test_get_settings_returns_one_cached_instance() -> None:
    first = config_module.get_settings()
    second = config_module.get_settings()

    assert first is second


def test_get_settings_is_isolated_from_the_module_level_alias() -> None:
    resolved = config_module.get_settings()
    resolved.log_level = "CRITICAL"

    assert config_module.settings.log_level == "INFO"


# --------------------------------------------------------------------------- #
# assignment validation
# --------------------------------------------------------------------------- #


def test_settings_validate_assignment_is_enabled() -> None:
    """Without it, ``settings.log_level = "trace"`` is silently accepted."""
    assert Settings.model_config["validate_assignment"] is True


def test_assignment_rejects_an_unknown_log_level() -> None:
    """The CLI writes ``settings.log_level`` directly, so the mutation is guarded."""
    settings = Settings(_env_file=None)

    with pytest.raises(ValidationError):
        settings.log_level = "trace"


def test_assignment_normalises_a_valid_log_level() -> None:
    settings = Settings(_env_file=None)

    settings.log_level = "debug"

    assert settings.log_level == "DEBUG"


@pytest.mark.parametrize("level", LOG_LEVELS)
def test_every_documented_level_is_accepted_on_assignment(level: str) -> None:
    settings = Settings(_env_file=None)

    settings.log_level = level.lower()

    assert settings.log_level == level


def test_a_rejected_assignment_leaves_the_previous_value_intact() -> None:
    """A failed assignment must not half-apply, or the CLI would run at the old level."""
    settings = Settings(_env_file=None, log_level="WARNING")

    with pytest.raises(ValidationError):
        settings.log_level = "trace"

    assert settings.log_level == "WARNING"


def test_assignment_rejects_a_non_integer_for_a_count_field() -> None:
    settings = Settings(_env_file=None)

    with pytest.raises(ValidationError):
        settings.max_parallel_tasks = "banana"


def test_assignment_accepts_a_valid_count_field() -> None:
    settings = Settings(_env_file=None)

    settings.max_cycles = 5

    assert settings.max_cycles == 5


def test_assignment_coerces_a_string_to_the_declared_path_type() -> None:
    settings = Settings(_env_file=None)

    settings.workspace = "/tmp"

    assert settings.workspace == Path("/tmp")


def test_an_unknown_provider_is_rejected_on_both_paths() -> None:
    """``provider`` is strict everywhere; the declared type is the protection.

    Nothing can put an unknown provider on a :class:`Settings`, so ``LLMClient``'s
    unsupported-provider branch is unreachable in production and stays a defensive
    guard, pinned from the outside in ``tests/test_llm.py``.
    """
    with pytest.raises(ValidationError):
        Settings(_env_file=None, provider="not-a-provider")

    settings = Settings(_env_file=None)

    with pytest.raises(ValidationError):
        settings.provider = "not-a-provider"


def test_a_real_provider_assignment_still_round_trips() -> None:
    """A recognised name is coerced to the enum, not stored as a bare string.

    :meth:`Settings.api_key_for` compares with ``is``; a stored ``'anthropic'``
    would fail that identity check and silently resolve to no API key.
    """
    settings = Settings(_env_file=None)

    settings.provider = "anthropic"

    assert settings.provider is Provider.ANTHROPIC


def test_a_provider_name_is_resolved_to_the_enum_on_assignment() -> None:
    """The field is declared ``Provider``; assignment must not weaken the type."""
    settings = Settings(_env_file=None)

    settings.provider = Provider.OPENAI.value

    assert settings.provider is Provider.OPENAI


def test_api_key_for_survives_a_provider_assigned_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The end-to-end consequence: an assigned name still finds the API key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    settings = Settings(_env_file=None, anthropic_api_key="from-settings")

    settings.provider = "anthropic"

    assert settings.api_key_for() == "from-settings"


# --------------------------------------------------------------------------- #
# unreadable configuration
# --------------------------------------------------------------------------- #


def _assert_actionable_configuration_error(excinfo: pytest.ExceptionInfo[Exception]) -> None:
    """Assert the operator is told what is wrong and where to look.

    Args:
        excinfo: The captured :class:`ConfigurationError`.
    """
    message = str(excinfo.value)
    assert "could not load its configuration" in message
    assert "FURROW_" in message
    assert ".env" in message
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, ValidationError)


def test_get_settings_explains_a_bad_env_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo in a ``FURROW_*`` variable must not surface as a bare traceback."""
    monkeypatch.setenv("FURROW_LOG_LEVEL", "LOUD")
    config_module.get_settings.cache_clear()

    with pytest.raises(ConfigurationError) as excinfo:
        config_module.get_settings()

    _assert_actionable_configuration_error(excinfo)
    assert "log_level='LOUD'" in str(excinfo.value)


def test_get_settings_explains_a_bad_dotenv_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``get_settings()`` reads ``.env``, and a bad entry there must be actionable too.

    The path is redirected with ``monkeypatch.chdir`` rather than by injecting a
    custom ``env_file``: ``Settings.model_config`` is the single declaration of
    the dotenv location, and honouring it the way the real code does is what makes
    this a faithful reproduction of ``cd <project> && import furrow``.
    """
    (tmp_path / ".env").write_text("FURROW_LOG_LEVEL=\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    config_module.get_settings.cache_clear()

    with pytest.raises(ConfigurationError) as excinfo:
        config_module.get_settings()

    _assert_actionable_configuration_error(excinfo)
    assert "log_level=''" in str(excinfo.value)
    assert str(tmp_path / ".env") in str(excinfo.value)


def _a_real_validation_error() -> ValidationError:
    """Return a genuine :class:`ValidationError` from a rejected ``log_level``.

    Built by actually failing validation rather than by hand-assembling error
    context, so the summary helpers are exercised on the real error shape.

    Returns:
        The captured error, with its ``__cause__`` chain intact.
    """
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, log_level="LOUD")

    return excinfo.value


def test_describe_invalid_settings_renders_field_equals_value() -> None:
    """The one-line summary is what makes the message actionable on its own."""
    assert config_module._describe_invalid_settings(_a_real_validation_error()) == (
        "log_level='LOUD'"
    )


def test_the_configuration_error_message_names_every_source() -> None:
    """Guards the message itself against an edit that drops a source or a hint."""
    message = config_module._configuration_error_message(_a_real_validation_error())

    assert "could not load its configuration" in message
    assert "FURROW_" in message
    assert ".env" in message
    assert "log_level='LOUD'" in message
    assert "chained" in message


def test_a_configuration_error_is_a_value_error() -> None:
    """Existing ``except ValueError`` handling of pydantic validation keeps working."""
    assert issubclass(ConfigurationError, ValueError)


def test_the_package_root_exports_the_public_config_api() -> None:
    """A caller must be able to ``except ConfigurationError`` off the root package.

    ``get_settings()`` runs at import time, so that one symbol is the first thing
    a user reaches for; the rest of the public config surface must be reachable
    from the root too, and named in ``__all__`` so the export is deliberate.
    """
    import furrow

    assert issubclass(furrow.ConfigurationError, ValueError)
    assert furrow.ConfigurationError is ConfigurationError
    assert furrow.LOG_LEVELS is LOG_LEVELS
    assert furrow.normalize_log_level is normalize_log_level
    assert furrow.configure_logging is configure_logging
    for name in (
        "ConfigurationError",
        "LOG_LEVELS",
        "normalize_log_level",
        "configure_logging",
    ):
        assert name in furrow.__all__, f"{name} missing from furrow.__all__"


def test_get_settings_still_builds_from_a_valid_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The wrap must not swallow a good configuration."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FURROW_LOG_LEVEL", "debug")
    config_module.get_settings.cache_clear()

    assert config_module.get_settings().log_level == "DEBUG"


def test_the_module_imports_with_a_valid_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``settings = get_settings()`` runs at module scope; a good dotenv must survive.

    ``runpy`` executes the real source in a throwaway namespace, so this is the
    import-time path itself without rebinding :data:`furrow.config.settings` and
    disturbing the shared singleton the rest of the suite relies on.
    """
    (tmp_path / ".env").write_text("FURROW_LOG_LEVEL=WARNING\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    namespace = runpy.run_path(str(Path(config_module.__file__)))

    assert namespace["settings"].log_level == "WARNING"


# --------------------------------------------------------------------------- #
# log levels
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _isolate_root_logging() -> Iterator[None]:
    """Snapshot and restore the process-wide root logger around every test.

    :func:`~furrow.config.configure_logging` installs a handler on the root logger
    when it has none and always re-applies the root *level*. Both are global, so a
    leaked level would make other tests' logging order-dependent and a leaked
    handler would accumulate for the rest of the session.

    Yields:
        None, for tests that only need the state snapshotted.
    """
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    try:
        yield
    finally:
        root.setLevel(saved_level)
        root.handlers[:] = saved_handlers


def test_log_levels_is_the_canonical_tuple() -> None:
    """The one spelling the validator, the CLI and the docs all read from."""
    assert LOG_LEVELS == ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("debug", "DEBUG"),
        ("  Info  ", "INFO"),
        ("critical", "CRITICAL"),
        ("WARNING", "WARNING"),
        ("  error ", "ERROR"),
    ],
)
def test_normalize_log_level_upper_cases_and_strips(raw: str, expected: str) -> None:
    assert normalize_log_level(raw) == expected


def test_normalize_log_level_rejects_an_unknown_level() -> None:
    with pytest.raises(ValueError):
        normalize_log_level("trace")


def test_the_rejection_message_names_every_allowed_level() -> None:
    with pytest.raises(ValueError) as excinfo:
        normalize_log_level("trace")

    message = str(excinfo.value)
    for level in LOG_LEVELS:
        assert level in message
    assert "trace" in message


def test_log_level_defaults_to_info() -> None:
    assert Settings(_env_file=None).log_level == "INFO"


def test_the_log_level_field_validator_normalises() -> None:
    assert Settings(_env_file=None, log_level="debug").log_level == "DEBUG"


def test_the_log_level_field_validator_rejects_an_unknown_level() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level="trace")


@pytest.mark.parametrize("level", LOG_LEVELS)
def test_every_documented_level_is_accepted(level: str) -> None:
    assert Settings(_env_file=None, log_level=level).log_level == level


@pytest.mark.parametrize("level", LOG_LEVELS)
def test_every_documented_level_is_accepted_in_any_case(level: str) -> None:
    assert Settings(_env_file=None, log_level=level.lower()).log_level == level


def test_the_log_level_env_override_is_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_LOG_LEVEL", "debug")

    assert Settings(_env_file=None).log_level == "DEBUG"


def test_configure_logging_returns_the_level_it_applied() -> None:
    assert configure_logging("  debug  ") == "DEBUG"


def test_configure_logging_sets_the_root_logger_level() -> None:
    configure_logging("DEBUG")

    assert logging.getLogger().level == logging.DEBUG

    configure_logging("ERROR")

    assert logging.getLogger().level == logging.ERROR


def test_configure_logging_is_idempotent() -> None:
    """Repeated calls are harmless: the CLI may configure logging more than once."""
    assert configure_logging("WARNING") == "WARNING"
    handlers = list(logging.getLogger().handlers)

    assert configure_logging("WARNING") == "WARNING"
    assert logging.getLogger().level == logging.WARNING
    assert logging.getLogger().handlers == handlers


def test_configure_logging_rejects_an_unknown_level() -> None:
    with pytest.raises(ValueError):
        configure_logging("trace")


def test_configure_logging_does_not_clobber_pre_existing_root_handlers() -> None:
    """``force=True`` would discard a host application's handlers; it is not used."""
    root = logging.getLogger()
    before = list(root.handlers)
    sentinel = logging.NullHandler()
    root.addHandler(sentinel)

    try:
        configure_logging("DEBUG")

        assert sentinel in root.handlers
        assert list(root.handlers) == [*before, sentinel]
    finally:
        root.removeHandler(sentinel)


def test_configure_logging_without_an_argument_uses_the_settings_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FURROW_LOG_LEVEL", "warning")

    assert configure_logging() == "WARNING"
    assert logging.getLogger().level == logging.WARNING


def test_an_explicit_argument_beats_the_settings_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FURROW_LOG_LEVEL", "warning")

    assert configure_logging("DEBUG") == "DEBUG"
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_honours_the_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``FURROW_LOG_LEVEL`` reaches the root logger through the real singleton."""
    monkeypatch.setenv("FURROW_LOG_LEVEL", "error")

    assert configure_logging() == "ERROR"
    assert logging.getLogger().level == logging.ERROR


def test_configure_logging_only_re_applies_the_root_level() -> None:
    """A child logger's own level is its own business; only root is re-levelled."""
    child = logging.getLogger("furrow.tests.only-root-is-levelled")
    child.setLevel(logging.CRITICAL)

    try:
        configure_logging("DEBUG")

        assert child.level == logging.CRITICAL
        assert logging.getLogger().level == logging.DEBUG
    finally:
        child.setLevel(logging.NOTSET)


# --------------------------------------------------------------------------- #
# .env.example
# --------------------------------------------------------------------------- #


def test_env_example_documents_every_setting_and_nothing_else() -> None:
    """A setting missing from the file is indistinguishable from an unset one."""
    assert set(_documented_env_examples()) == {
        f"FURROW_{name.upper()}" for name in Settings.model_fields
    }


def test_no_env_example_value_is_empty() -> None:
    """Uncommenting a bare ``FURROW_MODEL=`` sets ``""``, which breaks the LLM call."""
    empty = sorted(name for name, value in _documented_env_examples().items() if not value.strip())

    assert empty == []


def test_every_env_example_value_is_accepted_by_settings() -> None:
    """Each example must be runnable, so validating it keeps them honest."""
    overrides = {
        name[len("FURROW_") :].lower(): value for name, value in _documented_env_examples().items()
    }

    settings = Settings(_env_file=None, **overrides)

    assert settings.log_level in LOG_LEVELS
    assert settings.model
    assert str(settings.workspace)
