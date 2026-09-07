from unittest.mock import MagicMock, patch

import pytest
from furrow.config import Plan, TaskModel
from furrow.core.orchestrator import Orchestrator


def test_orchestrator_get_tasks():
    orchestrator = Orchestrator(goal="test")
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    orchestrator.current_plan = plan
    assert len(orchestrator._get_tasks()) == 1
    assert orchestrator._get_tasks()[0].description == "do thing"


def test_orchestrator_get_tasks_no_plan():
    orchestrator = Orchestrator(goal="test")
    assert orchestrator._get_tasks() == []


def test_orchestrator_is_done_no_tasks():
    orchestrator = Orchestrator(goal="test")
    orchestrator.current_plan = None
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_all_completed():
    orchestrator = Orchestrator(goal="test")
    orchestrator.current_plan = Plan(
        tasks=[TaskModel(id="1", description="do thing", status="completed")],
        rationale="ok",
    )
    orchestrator.last_test_passed = True
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_with_failures():
    orchestrator = Orchestrator(goal="test")
    orchestrator.current_plan = Plan(
        tasks=[TaskModel(id="1", description="do thing", status="failed")],
        rationale="ok",
    )
    orchestrator.last_test_passed = True
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_tests_failed():
    orchestrator = Orchestrator(goal="test")
    orchestrator.current_plan = Plan(
        tasks=[TaskModel(id="1", description="do thing", status="completed")],
        rationale="ok",
    )
    orchestrator.last_test_passed = False
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_tests_pending():
    orchestrator = Orchestrator(goal="test")
    orchestrator.current_plan = Plan(
        tasks=[TaskModel(id="1", description="do thing", status="completed")],
        rationale="ok",
    )
    orchestrator.last_test_passed = None
    assert orchestrator._is_done() is False


def test_orchestrator_max_cycles_attribute():
    orchestrator = Orchestrator(goal="test")
    assert orchestrator.settings.max_cycles == 0
    orchestrator.settings.max_cycles = 2
    assert orchestrator.settings.max_cycles == 2
