from __future__ import annotations

from furrow.config import Plan, RunResult, TaskModel, TestResult, Usage


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_usage_total_tokens():
    u = Usage(input_tokens=3, output_tokens=7)
    assert u.total_tokens == 10


def test_usage_default_all_zeros():
    u = Usage()
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.calls == 0
    assert u.cost_usd == 0.0
    assert u.total_tokens == 0


def test_usage_add_accumulates_and_returns_self():
    a = Usage(input_tokens=10, output_tokens=5, calls=2, cost_usd=0.25)
    b = Usage(input_tokens=3, output_tokens=7, calls=1, cost_usd=0.75)
    result = a.add(b)
    assert result is a
    assert a.input_tokens == 13
    assert a.output_tokens == 12
    assert a.calls == 3
    assert a.cost_usd == 1.0


def test_task_model_status_normalization():
    assert TaskModel(id="1", description="d", status="completed").status == "completed"
    assert TaskModel(id="1", description="d", status="UNKNOWN").status == "pending"
    assert TaskModel(id="1", description="d", status=None).status == "pending"
    assert TaskModel(id="1", description="d", status=123).status == "pending"
    assert TaskModel(id="1", description="d", status=" COMPLETED ").status == "completed"


def test_run_result_defaults():
    r = RunResult()
    assert r.passed is False
    assert r.cycles == 0
    assert r.tasks_completed == 0
    assert r.tasks_failed == 0
    assert r.tasks_blocked == 0
    assert r.tests_run == 0
    assert r.skipped is False
    assert r.files_changed == []
    assert isinstance(r.usage, Usage)


def test_plan_and_test_result_basic_construction():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"
    assert p.rationale == "ok"

    tr = TestResult(passed=True, summary="ok", failures=[])
    assert tr.passed is True
    assert tr.summary == "ok"
    assert tr.failures == []
