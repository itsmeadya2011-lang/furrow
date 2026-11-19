from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow.config import (
    Plan,
    Provider,
    RunResult,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
    Usage,
    WorkerResult,
)
from furrow.costs import ModelPricing


def _settings(**overrides: object) -> Settings:
    """Build Settings from explicit values only, ignoring any local .env file."""
    return Settings(_env_file=None, **overrides)


# ---------------------------------------------------------------------------
# TaskModel
# ---------------------------------------------------------------------------


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.id == "1"
    assert t.description == "do thing"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None
    assert t.error is None
    assert t.attempts == 0
    assert t.started_at is None
    assert t.finished_at is None
    assert t.files_changed == []


def test_task_model_default_lists_are_not_shared():
    a = TaskModel(id="1", description="a")
    b = TaskModel(id="2", description="b")
    a.files.append("f.py")
    a.dependencies.append("0")
    a.files_changed.append("g.py")
    assert b.files == []
    assert b.dependencies == []
    assert b.files_changed == []


def test_task_model_accepts_all_fields():
    t = TaskModel(
        id="7",
        description="refactor",
        files=["a.py", "b.py"],
        dependencies=["1", "2"],
        status="running",
        result="partial",
        error="boom",
        attempts=2,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:05:00Z",
        files_changed=["a.py"],
    )
    assert t.id == "7"
    assert t.files == ["a.py", "b.py"]
    assert t.dependencies == ["1", "2"]
    assert t.status == "running"
    assert t.result == "partial"
    assert t.error == "boom"
    assert t.attempts == 2
    assert t.started_at == "2026-01-01T00:00:00Z"
    assert t.finished_at == "2026-01-01T00:05:00Z"
    assert t.files_changed == ["a.py"]


@pytest.mark.parametrize("status", sorted({s.value for s in TaskStatus}))
def test_task_model_accepts_every_known_status(status: str):
    t = TaskModel(id="1", description="d", status=status)
    assert t.status == status


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PENDING", "pending"),
        ("Running", "running"),
        ("COMPLETED", "completed"),
        ("Failed", "failed"),
        ("BLOCKED", "blocked"),
        ("Skipped", "skipped"),
        ("  running  ", "running"),
        ("\tcompleted\n", "completed"),
    ],
)
def test_task_model_normalizes_case_and_whitespace(raw: str, expected: str):
    t = TaskModel(id="1", description="d", status=raw)
    assert t.status == expected


@pytest.mark.parametrize(
    "raw",
    ["unknown", "done", "", "  ", "in-progress", "cancelled", "PENDING!"],
)
def test_task_model_unknown_status_becomes_pending(raw: str):
    t = TaskModel(id="1", description="d", status=raw)
    assert t.status == "pending"


@pytest.mark.parametrize("raw", [None, 123, 1.5, True, [], {}, ("pending",)])
def test_task_model_non_string_status_becomes_pending(raw: object):
    t = TaskModel(id="1", description="d", status=raw)
    assert t.status == "pending"


def test_task_model_status_is_plain_str():
    t = TaskModel(id="1", description="d")
    assert type(t.status) is str
    assert t.status == TaskStatus.PENDING


def test_task_model_status_enum_is_coerced_by_name():
    t = TaskModel(id="1", description="d", status=TaskStatus.FAILED)
    assert t.status == "failed"


@pytest.mark.parametrize("missing", ["id", "description"])
def test_task_model_requires_id_and_description(missing: str):
    kwargs = {"id": "1", "description": "d"}
    del kwargs[missing]
    with pytest.raises(ValidationError):
        TaskModel(**kwargs)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_plan_construction_keeps_all_tasks_in_order():
    tasks = [
        TaskModel(id="1", description="first"),
        TaskModel(id="2", description="second", dependencies=["1"]),
    ]
    p = Plan(tasks=tasks, rationale="because")
    assert [t.id for t in p.tasks] == ["1", "2"]
    assert p.tasks[1].dependencies == ["1"]
    assert p.rationale == "because"


def test_plan_requires_rationale():
    with pytest.raises(ValidationError):
        Plan(tasks=[])


def test_plan_requires_tasks():
    with pytest.raises(ValidationError):
        Plan(rationale="only rationale")


def test_plan_accepts_empty_tasks():
    p = Plan(tasks=[], rationale="nothing to do")
    assert p.tasks == []
    assert p.rationale == "nothing to do"


# ---------------------------------------------------------------------------
# TestResult
# ---------------------------------------------------------------------------


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_test_result_defaults():
    t = TestResult(passed=True, summary="ok")
    assert t.passed is True
    assert t.summary == "ok"
    assert t.failures == []
    assert t.skipped is False


def test_test_result_failures_are_kept():
    t = TestResult(passed=False, summary="2 failed", failures=["a", "b"])
    assert t.passed is False
    assert t.summary == "2 failed"
    assert t.failures == ["a", "b"]


def test_test_result_default_failures_list_is_not_shared():
    a = TestResult(passed=True, summary="a")
    b = TestResult(passed=True, summary="b")
    a.failures.append("only in a")
    assert b.failures == []


def test_test_result_skipped_flag_default_and_override():
    assert TestResult(passed=True, summary="ok").skipped is False
    assert TestResult(passed=True, summary="ok", skipped=True).skipped is True


def test_test_result_is_not_collected_by_pytest():
    assert TestResult.__test__ is False


@pytest.mark.parametrize("missing", ["passed", "summary"])
def test_test_result_requires_passed_and_summary(missing: str):
    kwargs = {"passed": True, "summary": "ok"}
    del kwargs[missing]
    with pytest.raises(ValidationError):
        TestResult(**kwargs)


# ---------------------------------------------------------------------------
# WorkerResult
# ---------------------------------------------------------------------------


def test_worker_result_defaults():
    w = WorkerResult()
    assert w.summary == ""
    assert w.success is True
    assert w.files_changed == []
    assert w.error is None
    assert w.iterations == 0


def test_worker_result_success_with_details():
    w = WorkerResult(
        summary="implemented",
        success=True,
        files_changed=["a.py", "b.py"],
        iterations=3,
    )
    assert w.summary == "implemented"
    assert w.success is True
    assert w.files_changed == ["a.py", "b.py"]
    assert w.error is None
    assert w.iterations == 3


def test_worker_result_failure_carries_error():
    w = WorkerResult(summary="broke", success=False, error="tests failed", iterations=1)
    assert w.success is False
    assert w.error == "tests failed"
    assert w.iterations == 1


def test_worker_result_default_files_list_is_not_shared():
    a = WorkerResult()
    b = WorkerResult()
    a.files_changed.append("a.py")
    assert b.files_changed == []


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


def test_usage_defaults():
    u = Usage()
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.calls == 0
    assert u.cost_usd == 0.0
    assert u.total_tokens == 0


def test_usage_total_tokens_sums_input_and_output():
    u = Usage(input_tokens=10, output_tokens=5)
    assert u.total_tokens == 15
    assert u.cost_usd == 0.0
    assert u.calls == 0


def test_usage_add_accumulates_every_field():
    a = Usage(input_tokens=10, output_tokens=5, calls=2, cost_usd=0.25)
    b = Usage(input_tokens=3, output_tokens=7, calls=1, cost_usd=0.75)
    result = a.add(b)
    assert result is a
    assert a.input_tokens == 13
    assert a.output_tokens == 12
    assert a.calls == 3
    assert a.cost_usd == 1.0
    assert a.total_tokens == 25


def test_usage_add_zero_is_a_no_op():
    a = Usage(input_tokens=4, output_tokens=6, calls=1, cost_usd=0.5)
    a.add(Usage())
    assert a.input_tokens == 4
    assert a.output_tokens == 6
    assert a.calls == 1
    assert a.cost_usd == 0.5


def test_usage_add_is_chainable():
    u = (
        Usage()
        .add(Usage(input_tokens=1, output_tokens=2, calls=1))
        .add(Usage(input_tokens=3, output_tokens=4, calls=1))
    )
    assert u.input_tokens == 4
    assert u.output_tokens == 6
    assert u.calls == 2
    assert u.total_tokens == 10


def test_usage_add_does_not_mutate_argument():
    a = Usage(input_tokens=1, output_tokens=1, calls=1, cost_usd=0.1)
    b = Usage(input_tokens=2, output_tokens=3, calls=4, cost_usd=0.2)
    a.add(b)
    assert b.input_tokens == 2
    assert b.output_tokens == 3
    assert b.calls == 4
    assert b.cost_usd == 0.2


def test_usage_instances_are_independent():
    a = Usage(input_tokens=5, output_tokens=5, calls=1, cost_usd=1.0)
    b = Usage()
    assert b.total_tokens == 0
    assert b.cost_usd == 0.0
    assert a.total_tokens == 10


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------


def test_run_result_defaults():
    r = RunResult()
    assert r.passed is False
    assert r.cycles == 0
    assert r.tasks_completed == 0
    assert r.tasks_failed == 0
    assert r.tasks_blocked == 0
    assert r.tests_run == 0
    assert r.summary == ""
    assert r.stopped_reason == ""
    assert r.skipped is False
    assert isinstance(r.usage, Usage)
    assert r.usage.total_tokens == 0
    assert r.files_changed == []


def test_run_result_default_usage_and_files_are_not_shared():
    a = RunResult()
    b = RunResult()
    a.usage.add(Usage(input_tokens=9, output_tokens=1, calls=1))
    a.files_changed.append("a.py")
    assert b.usage.total_tokens == 0
    assert b.files_changed == []


def test_run_result_accepts_all_fields():
    r = RunResult(
        passed=True,
        cycles=3,
        tasks_completed=4,
        tasks_failed=1,
        tasks_blocked=2,
        tests_run=12,
        summary="done",
        stopped_reason="max_cycles",
        skipped=False,
        usage=Usage(input_tokens=10, output_tokens=20, calls=2, cost_usd=0.5),
        files_changed=["a.py", "b.py"],
    )
    assert r.passed is True
    assert r.cycles == 3
    assert r.tasks_completed == 4
    assert r.tasks_failed == 1
    assert r.tasks_blocked == 2
    assert r.tests_run == 12
    assert r.summary == "done"
    assert r.stopped_reason == "max_cycles"
    assert r.skipped is False
    assert r.usage.total_tokens == 30
    assert r.usage.cost_usd == 0.5
    assert r.files_changed == ["a.py", "b.py"]


def test_run_result_usage_can_be_accumulated_in_place():
    r = RunResult(passed=True)
    r.usage.add(Usage(input_tokens=1, output_tokens=2, calls=1))
    r.usage.add(Usage(input_tokens=3, output_tokens=4, calls=1))
    assert r.usage.total_tokens == 10
    assert r.usage.calls == 2


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


def test_provider_values():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_provider_members():
    assert {p.value for p in Provider} == {"anthropic", "openai", "ollama"}


def test_provider_is_str_enum():
    assert isinstance(Provider.ANTHROPIC, str)
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider("openai") is Provider.OPENAI
    with pytest.raises(ValueError):
        Provider("unknown")


# ---------------------------------------------------------------------------
# TaskStatus
# ---------------------------------------------------------------------------


def test_task_status_enum():
    assert TaskStatus.PENDING == "pending"
    assert TaskStatus.RUNNING == "running"
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus.FAILED == "failed"
    assert TaskStatus.BLOCKED == "blocked"
    assert TaskStatus.SKIPPED == "skipped"


def test_task_status_members_and_lookup():
    assert {s.value for s in TaskStatus} == {
        "pending",
        "running",
        "completed",
        "failed",
        "blocked",
        "skipped",
    }
    assert isinstance(TaskStatus.PENDING, str)
    assert TaskStatus("running") is TaskStatus.RUNNING
    with pytest.raises(ValueError):
        TaskStatus("unknown")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_defaults():
    s = _settings()
    assert s.provider is Provider.ANTHROPIC
    assert s.model == "claude-sonnet-4-20250514"
    assert s.planner_model == "claude-3-5-haiku-20241022"
    assert s.worker_model == "claude-3-5-sonnet-20241022"
    assert s.tester_model == "claude-3-5-sonnet-20241022"
    assert s.anthropic_api_key is None
    assert s.openai_api_key is None
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"
    assert s.max_worker_iterations == 12
    assert s.max_retries == 3
    assert s.max_task_retries == 1
    assert s.request_timeout == 120.0
    assert s.command_timeout == 300.0
    assert s.test_timeout == 600.0
    assert s.max_tokens == 4096
    assert s.state_file == Path(".furrow/state.json")
    assert s.model_pricing == {}
    assert isinstance(s.workspace, Path)


def test_settings_workspace_defaults_to_cwd():
    s = _settings()
    assert s.workspace == Path.cwd()


def test_settings_accepts_overrides():
    s = _settings(
        provider="openai",
        model="gpt-4o",
        max_parallel_tasks=2,
        max_cycles=10,
        log_level="DEBUG",
    )
    assert s.provider is Provider.OPENAI
    assert s.model == "gpt-4o"
    assert s.max_parallel_tasks == 2
    assert s.max_cycles == 10
    assert s.log_level == "DEBUG"


def test_settings_state_path_with_explicit_workspace():
    s = _settings()
    assert s.state_path(Path("/tmp/ws")) == Path("/tmp/ws/.furrow/state.json")


def test_settings_state_path_uses_workspace_when_none_passed():
    s = _settings(workspace=Path("/my/workspace"))
    assert s.state_path() == Path("/my/workspace/.furrow/state.json")


def test_settings_state_path_with_custom_state_file():
    s = _settings(state_file=Path("custom/state.json"))
    assert s.state_path(Path("/tmp/ws")) == Path("/tmp/ws/custom/state.json")


def test_settings_state_path_explicit_workspace_beats_setting():
    s = _settings(workspace=Path("/from/settings"))
    assert s.state_path(Path("/explicit")) == Path("/explicit/.furrow/state.json")


def test_settings_state_path_accepts_string_workspace():
    s = _settings()
    assert s.state_path("/tmp/ws") == Path("/tmp/ws/.furrow/state.json")


def test_settings_state_path_with_absolute_state_file():
    s = _settings(state_file=Path("/var/lib/furrow/state.json"))
    assert s.state_path(Path("/tmp/ws")) == Path("/var/lib/furrow/state.json")


def test_settings_state_path_returns_path(tmp_path: Path):
    s = _settings()
    assert isinstance(s.state_path(tmp_path), Path)
    assert s.state_path(tmp_path) == tmp_path / ".furrow" / "state.json"


def test_settings_pricing_overrides_empty():
    s = _settings()
    assert s.pricing_overrides() == {}


def test_settings_pricing_overrides_returns_copy():
    pricing = {"my-model": ModelPricing(input_per_million=1.0, output_per_million=2.0)}
    s = _settings(model_pricing=pricing)
    overrides = s.pricing_overrides()
    assert overrides == pricing
    overrides["my-model"] = ModelPricing(input_per_million=9.0, output_per_million=9.0)
    assert s.model_pricing["my-model"].input_per_million == 1.0


def test_settings_pricing_overrides_accepts_plain_dicts():
    s = _settings(model_pricing={"my-model": {"input_per_million": 1.5, "output_per_million": 2.5}})
    assert s.model_pricing["my-model"].input_per_million == 1.5
    assert s.model_pricing["my-model"].output_per_million == 2.5
    assert isinstance(s.model_pricing["my-model"], ModelPricing)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("max_parallel_tasks", 0),
        ("max_parallel_tasks", -1),
        ("max_worker_iterations", 0),
        ("max_worker_iterations", -5),
        ("max_tokens", 0),
        ("max_tokens", -100),
        ("max_cycles", -1),
        ("max_retries", -1),
        ("max_task_retries", -1),
        ("request_timeout", 0),
        ("request_timeout", -1.0),
        ("command_timeout", 0),
        ("command_timeout", -1.0),
        ("test_timeout", 0),
        ("test_timeout", -1.0),
    ],
)
def test_settings_rejects_out_of_bounds_values(field: str, bad_value: object):
    with pytest.raises(ValidationError):
        _settings(**{field: bad_value})


@pytest.mark.parametrize(
    ("field", "min_value"),
    [
        ("max_parallel_tasks", 1),
        ("max_worker_iterations", 1),
        ("max_tokens", 1),
        ("max_cycles", 0),
        ("max_retries", 0),
        ("max_task_retries", 0),
        ("request_timeout", 0.001),
        ("command_timeout", 0.001),
        ("test_timeout", 0.001),
    ],
)
def test_settings_accepts_lower_bounds(field: str, min_value: object):
    s = _settings(**{field: min_value})
    assert getattr(s, field) == min_value


def test_settings_accepts_large_values():
    s = _settings(
        max_parallel_tasks=100,
        max_cycles=1000,
        max_worker_iterations=500,
        max_tokens=1_000_000,
        request_timeout=3600.0,
    )
    assert s.max_parallel_tasks == 100
    assert s.max_cycles == 1000
    assert s.max_worker_iterations == 500
    assert s.max_tokens == 1_000_000
    assert s.request_timeout == 3600.0


def test_settings_env_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FURROW_MAX_CYCLES", "7")
    monkeypatch.setenv("FURROW_LOG_LEVEL", "WARNING")
    s = _settings()
    assert s.max_cycles == 7
    assert s.log_level == "WARNING"
