"""Configuration contract: construction, ``FURROW_`` env binding, and the documented defaults.

The README's ``FURROW_*`` table is asserted against a freshly built :class:`Settings` so the
documentation cannot drift away from the code without a test going red.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow.config import Plan, Provider, Settings, TaskModel
from furrow.config import TestResult as Verdict

#: Defaults as spelled out in README.md's "Configuration" table, for fields whose
#: documented value is a literal string rather than a number.
DOCUMENTED_DEFAULTS: dict[str, object] = {
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514",
    "planner_model": "claude-3-5-haiku-20241022",
    "worker_model": "claude-3-5-sonnet-20241022",
    "tester_model": "claude-3-5-sonnet-20241022",
    "ollama_base_url": "http://localhost:11434",
    "log_level": "INFO",
    "anthropic_api_key": "unset",
    "openai_api_key": "unset",
    "workspace": "current working directory",
}

#: Numeric defaults, compared by value so ``1.0`` and ``1`` both count.
DOCUMENTED_NUMERIC_DEFAULTS: dict[str, float] = {
    "max_cycles": 0,
    "max_parallel_tasks": 5,
    "llm_max_retries": 3,
    "llm_backoff_seconds": 1.0,
    "test_timeout_seconds": 120,
}

README_DEFAULT_ROW = re.compile(r"^\|\s*`FURROW_([A-Z0-9_]+)`\s*\|(.*)$")


def readme_settings_table(readme: Path) -> dict[str, str]:
    """Extract ``{field: documented default}`` from the README's configuration table."""
    rows: dict[str, str] = {}
    for line in readme.read_text(encoding="utf-8").splitlines():
        match = README_DEFAULT_ROW.match(line)
        if match is not None:
            rows[match.group(1).lower()] = match.group(2).split("|")[0].strip().strip("`")
    return rows


def test_settings_constructs_with_no_configuration() -> None:
    settings = Settings()

    assert settings.provider is Provider.ANTHROPIC
    assert settings.model == "claude-sonnet-4-20250514"
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0
    assert settings.test_timeout_seconds == 120


@pytest.mark.parametrize(("field", "expected"), sorted(DOCUMENTED_DEFAULTS.items()))
def test_documented_string_defaults(field: str, expected: object) -> None:
    settings = Settings()

    if expected == "unset":
        assert getattr(settings, field) is None
    elif expected == "current working directory":
        assert getattr(settings, field) == Path.cwd()
    else:
        assert getattr(settings, field) == expected


@pytest.mark.parametrize(("field", "expected"), sorted(DOCUMENTED_NUMERIC_DEFAULTS.items()))
def test_documented_numeric_defaults(field: str, expected: float) -> None:
    value = getattr(Settings(), field)

    assert float(value) == pytest.approx(expected)


def test_max_cycles_zero_is_the_documented_unlimited_default() -> None:
    # 0 is the sentinel the orchestrator's `max_cycles > 0` guard tests; a Settings default
    # of 0 is therefore what "0 means unlimited" means in practice.
    assert Settings().max_cycles == 0

    settings = Settings(max_cycles=0)

    assert settings.max_cycles == 0


def test_readme_table_lists_exactly_the_settings_fields(repo_root: Path) -> None:
    documented = readme_settings_table(repo_root / "README.md")

    assert set(documented) == set(Settings.model_fields)


def test_readme_table_defaults_match_a_fresh_settings(repo_root: Path) -> None:
    documented = readme_settings_table(repo_root / "README.md")
    settings = Settings()

    for field, text in documented.items():
        value = getattr(settings, field)
        if value is None:
            assert text == "unset", field
        elif isinstance(value, Provider):
            assert text == value.value, field
        elif isinstance(value, Path):
            assert text == "current working directory", field
        elif isinstance(value, bool):
            assert text == str(value), field
        elif isinstance(value, (int, float)):
            assert float(text) == pytest.approx(float(value)), field
        else:
            assert text == str(value), field


@pytest.mark.parametrize(
    ("variable", "field", "raw", "expected"),
    [
        ("FURROW_PROVIDER", "provider", "ollama", Provider.OLLAMA),
        ("FURROW_MODEL", "model", "gpt-test", "gpt-test"),
        ("FURROW_PLANNER_MODEL", "planner_model", "planner-x", "planner-x"),
        ("FURROW_WORKER_MODEL", "worker_model", "worker-x", "worker-x"),
        ("FURROW_TESTER_MODEL", "tester_model", "tester-x", "tester-x"),
        ("FURROW_ANTHROPIC_API_KEY", "anthropic_api_key", "sk-ant-test", "sk-ant-test"),
        ("FURROW_OPENAI_API_KEY", "openai_api_key", "sk-openai-test", "sk-openai-test"),
        ("FURROW_OLLAMA_BASE_URL", "ollama_base_url", "http://box:1234", "http://box:1234"),
        ("FURROW_MAX_CYCLES", "max_cycles", "4", 4),
        ("FURROW_MAX_PARALLEL_TASKS", "max_parallel_tasks", "9", 9),
        ("FURROW_LLM_MAX_RETRIES", "llm_max_retries", "1", 1),
        ("FURROW_LLM_BACKOFF_SECONDS", "llm_backoff_seconds", "0.25", 0.25),
        ("FURROW_TEST_TIMEOUT_SECONDS", "test_timeout_seconds", "7", 7),
        ("FURROW_LOG_LEVEL", "log_level", "DEBUG", "DEBUG"),
        ("FURROW_WORKSPACE", "workspace", "/tmp", Path("/tmp")),
    ],
)
def test_furrow_prefixed_env_vars_bind(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    field: str,
    raw: str,
    expected: object,
) -> None:
    monkeypatch.setenv(variable, raw)

    assert getattr(Settings(), field) == expected


def test_env_prefix_ignores_unprefixed_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_CYCLES", "7")
    monkeypatch.setenv("FURROW_MODEL", "prefixed-wins")

    settings = Settings()

    assert settings.max_cycles == 0
    assert settings.model == "prefixed-wins"


def test_dotenv_file_is_read_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("FURROW_MAX_CYCLES=6\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert Settings().max_cycles == 6


def test_unknown_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_PROVIDER", "not-a-provider")

    with pytest.raises(ValidationError):
        Settings()


def test_provider_values_are_the_documented_triple() -> None:
    assert [provider.value for provider in Provider] == ["anthropic", "openai", "ollama"]


def test_schema_models_round_trip() -> None:
    task = TaskModel(id="1", description="do the thing")

    assert task.status == "pending"
    assert task.files == []
    assert task.dependencies == []
    assert task.result is None

    plan = Plan(tasks=[task], rationale="ok")
    assert plan.tasks[0].description == "do the thing"

    result = Verdict(passed=True, summary="ok")
    assert result.passed is True
    assert result.failures == []


def test_settings_constructor_overrides_beat_defaults() -> None:
    settings = Settings(provider=Provider.OPENAI, max_cycles=3, max_parallel_tasks=2)

    assert settings.provider is Provider.OPENAI
    assert settings.max_cycles == 3
    assert settings.max_parallel_tasks == 2


def test_module_level_singleton_is_a_settings_instance() -> None:
    from furrow.config import settings

    assert isinstance(settings, Settings)
