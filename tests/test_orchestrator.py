import pytest
from unittest.mock import AsyncMock

from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient
from furrow.config import settings, TaskModel


def test_orchestrator_initial_state():
    stub_client = AsyncMock(spec=LLMClient)
    orchestrator = Orchestrator(goal="x", client=stub_client)
    assert orchestrator.original_goal == "x"
    assert orchestrator.tasks == []
    assert orchestrator.cycles == 0


def test_is_done_empty_tasks():
    stub_client = AsyncMock(spec=LLMClient)
    orchestrator = Orchestrator(goal="x", client=stub_client)
    assert orchestrator._is_done() is False


def test_is_done_with_failed_task():
    stub_client = AsyncMock(spec=LLMClient)
    orchestrator = Orchestrator(goal="x", client=stub_client)
    task = TaskModel(id="1", description="do thing", status="failed")
    orchestrator.tasks = [task]
    assert orchestrator._is_done() is False


def test_is_done_all_completed_no_new_work():
    stub_client = AsyncMock(spec=LLMClient)
    orchestrator = Orchestrator(goal="x", client=stub_client)
    task = TaskModel(id="1", description="do thing", status="completed")
    orchestrator.tasks = [task]
    orchestrator.planner_produced_new_work = False
    assert orchestrator._is_done() is True


def test_model_override_does_not_mutate_global_settings():
    global_model_before = settings.model
    stub_client = AsyncMock(spec=LLMClient)
    orchestrator = Orchestrator(goal="x", client=stub_client, model_override="gpt-4")
    assert orchestrator.settings.model == "gpt-4"
    assert settings.model == global_model_before
