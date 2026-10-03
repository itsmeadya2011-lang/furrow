"""Smoke coverage for the pydantic models in :mod:`furrow.config`.

These are construction checks only: they prove the models build and expose the fields the
rest of the code reads, nothing more. The real regression coverage lives in
``test_orchestrator.py`` (the loop: cycle counting, dependency scheduling, the stop
conditions and ``RunOutcome``), ``test_llm.py`` (the provider client) and ``test_agents.py``
(the planner/worker/tester agents). The ``PytestCollectionWarning`` that ``TestResult``
provokes -- pytest collects any visible class whose name starts with ``Test`` -- is handled
once, centrally, in ``conftest.py`` via ``TestResult.__test__ = False``; that is also why no
test module needs a local marker for it.
"""

from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True
