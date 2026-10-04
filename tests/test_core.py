from furrow.config import Plan, TaskModel
from furrow.config import TestResult as ResultModel


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = ResultModel(passed=True, summary="ok", failures=[])
    assert t.passed is True
