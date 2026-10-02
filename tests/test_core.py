from furrow.config import Plan, TaskModel
from furrow.config import TestResult as _TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    # Imported under an alias on purpose: pytest collects every module-level name
    # that starts with ``Test``, so binding the model as ``TestResult`` produced a
    # PytestCollectionWarning. The alias keeps the class out of collection
    # without touching production code.
    t = _TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True
