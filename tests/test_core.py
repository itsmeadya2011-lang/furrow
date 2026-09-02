import pytest
from furrow.config import Plan, TaskModel, TestResult, Provider


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_provider_ollama():
    assert Provider.OLLAMA is not None
    assert Provider.OLLAMA == "ollama"
