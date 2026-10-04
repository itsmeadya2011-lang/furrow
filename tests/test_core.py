from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"
    assert p.rationale == "ok"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True
    assert t.summary == "ok"
