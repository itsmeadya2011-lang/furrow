from __future__ import annotations

from pathlib import Path

import pytest

from furrow.config import (
    Plan,
    Provider,
    RunResult,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
    WorkerResult,
)
from furrow.costs import ModelPricing


def test_task_model_defaults():
    t = TaskModel(id="1", description="d")
    assert t.status == "pending"
    assert t.attempts == 0
    assert t.error is None
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None
    assert t.started_at is None
    assert t.finished_at is None


def test_task_status_enum():
    assert TaskStatus.PENDING == "pending"
    assert TaskStatus.RUNNING == "running"
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus.FAILED == "failed"
    assert TaskStatus.BLOCKED == "blocked"
    assert TaskStatus.SKIPPED == "skipped"


def test_worker_result_defaults():
    w = WorkerResult()
    assert w.summary == ""
    assert w.success is True
    assert w.files_changed == []
    assert w.error is None
    assert w.iterations == 0


def test_run_result_defaults():
    r = RunResult()
    assert r.passed is False
    assert r.cycles == 0
    assert r.tasks_completed == 0
    assert r.tasks_failed == 0
    assert r.summary == ""
    assert r.stopped_reason == ""
    assert r.skipped is False


def test_plan_and_test_result_still_work():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"
    assert p.rationale == "ok"

    tr = TestResult(passed=True, summary="ok", failures=[])
    assert tr.passed is True
    assert tr.summary == "ok"
    assert tr.failures == []


def test_settings_state_path_absolute_workspace():
    s = Settings()
    assert s.state_path(Path("/tmp/ws")) == Path("/tmp/ws/.furrow/state.json")


def test_settings_state_path_custom_state_file():
    s = Settings(state_file=Path("custom/state.json"))
    assert s.state_path(Path("/tmp/ws")) == Path("/tmp/ws/custom/state.json")


def test_settings_state_path_no_argument_uses_workspace():
    s = Settings(workspace=Path("/my/workspace"))
    assert s.state_path() == Path("/my/workspace/.furrow/state.json")


def test_test_result_not_collected_by_pytest():
    assert TestResult.__test__ is False
    assert set(TestResult.model_fields) == {"passed", "summary", "failures", "skipped"}


def test_skipped_defaults():
    tr = TestResult(passed=True, summary="ok")
    assert tr.skipped is False
    r = RunResult()
    assert r.skipped is False


def test_skipped_can_be_set():
    tr = TestResult(passed=True, summary="ok", skipped=True)
    assert tr.skipped is True
    r = RunResult(passed=True, skipped=True)
    assert r.skipped is True


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FURROW_MAX_CYCLES", "7")
    s = Settings(_env_file=None)
    assert s.max_cycles == 7


def test_settings_provider_default():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC


def test_settings_per_agent_provider_defaults_to_none():
    s = Settings()
    assert s.planner_provider is None
    assert s.worker_provider is None
    assert s.tester_provider is None


def test_settings_per_agent_provider_can_be_overridden(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FURROW_WORKER_PROVIDER", "openai")
    s = Settings(_env_file=None)
    assert s.worker_provider == Provider.OPENAI


def test_settings_per_agent_provider_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FURROW_PLANNER_PROVIDER", "ollama")
    s = Settings(_env_file=None)
    assert s.planner_provider == Provider.OLLAMA


def test_run_result_new_defaults():
    r = RunResult()
    assert r.tasks_blocked == 0
    assert r.tests_run == 0


def test_settings_new_timeouts_defaults():
    s = Settings()
    assert s.command_timeout == 300.0
    assert s.test_timeout == 600.0


def test_runstate_has_tests_run_field_defaulting_to_zero():
    from furrow.core.state import RunState

    state = RunState()
    assert state.tests_run == 0


def test_unknown_status_normalized_and_plan_parsing_succeeds():
    from furrow.agents.parsing import parse_plan

    data = {
        "rationale": "r",
        "tasks": [{"id": "t1", "description": "d", "status": "UNKNOWN"}],
    }
    plan = parse_plan(__import__("json").dumps(data))
    assert plan.tasks[0].status == "pending"


def test_numeric_settings_reject_invalid_values():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(max_parallel_tasks=0)
    with pytest.raises(ValidationError):
        Settings(max_worker_iterations=0)
    with pytest.raises(ValidationError):
        Settings(max_tokens=0)
    with pytest.raises(ValidationError):
        Settings(request_timeout=0)
    with pytest.raises(ValidationError):
        Settings(command_timeout=0)
    with pytest.raises(ValidationError):
        Settings(test_timeout=0)
    with pytest.raises(ValidationError):
        Settings(max_cycles=-1)
    with pytest.raises(ValidationError):
        Settings(max_retries=-1)
    with pytest.raises(ValidationError):
        Settings(max_task_retries=-1)


def test_numeric_settings_allow_valid_bounds():
    s = Settings(
        max_parallel_tasks=1,
        max_cycles=0,
        max_retries=0,
        max_task_retries=0,
        max_worker_iterations=1,
        max_tokens=1,
        request_timeout=0.001,
        command_timeout=0.001,
        test_timeout=0.001,
    )
    assert s.max_cycles == 0
    assert s.max_retries == 0
    assert s.max_task_retries == 0


def test_usage_cost_usd_defaults_to_zero():
    from furrow.config import Usage

    u = Usage()
    assert u.cost_usd == 0.0
    assert u.total_tokens == 0


def test_usage_add_accumulates_cost_usd():
    from furrow.config import Usage

    a = Usage(input_tokens=10, output_tokens=5, calls=2, cost_usd=0.25)
    b = Usage(input_tokens=3, output_tokens=7, calls=1, cost_usd=0.75)
    result = a.add(b)
    assert result is a
    assert a.cost_usd == 1.0
    assert a.input_tokens == 13
    assert a.output_tokens == 12
    assert a.calls == 3
    assert a.total_tokens == 25


def test_settings_model_pricing_defaults_to_empty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FURROW_MODEL_PRICING", raising=False)
    s = Settings(_env_file=None)
    assert s.model_pricing == {}
    assert s.pricing_overrides() == {}


def test_settings_accepts_model_pricing_as_a_dict():
    s = Settings(
        model_pricing={
            "my-model": ModelPricing(input_per_million=1.0, output_per_million=2.0)
        }
    )
    assert s.model_pricing["my-model"].input_per_million == 1.0
    assert s.model_pricing["my-model"].output_per_million == 2.0
    assert s.pricing_overrides() == {
        "my-model": ModelPricing(input_per_million=1.0, output_per_million=2.0)
    }


def test_settings_accepts_model_pricing_as_a_plain_dict():
    s = Settings(
        model_pricing={"my-model": {"input_per_million": 1.0, "output_per_million": 2.0}}
    )
    assert s.model_pricing["my-model"].input_per_million == 1.0
    assert s.model_pricing["my-model"].output_per_million == 2.0


def test_settings_reads_model_pricing_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(
        "FURROW_MODEL_PRICING",
        '{"my-model":{"input_per_million":1.0,"output_per_million":2.0}}',
    )
    s = Settings(_env_file=None)
    assert s.model_pricing["my-model"].input_per_million == 1.0
    assert s.model_pricing["my-model"].output_per_million == 2.0
    assert s.pricing_overrides() == {
        "my-model": ModelPricing(input_per_million=1.0, output_per_million=2.0)
    }


def test_pricing_overrides_returns_a_copy():
    s = Settings(
        model_pricing={
            "my-model": ModelPricing(input_per_million=1.0, output_per_million=2.0)
        }
    )
    overrides = s.pricing_overrides()
    overrides["my-model"] = ModelPricing(input_per_million=9.0, output_per_million=9.0)
    assert s.model_pricing["my-model"].input_per_million == 1.0

