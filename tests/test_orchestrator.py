from __future__ import annotations

import pytest

from furrow.agents.utils import clean_json_response
from furrow.config import Plan, TaskModel
from furrow.core.orchestrator import Orchestrator
from tests.conftest import MockLLMClient


class TestPlannerJSONParsing:
    def test_clean_json_no_fences(self):
        raw = '{"tasks": [], "rationale": "ok"}'
        assert clean_json_response(raw) == raw

    def test_clean_json_with_fences(self):
        raw = '```json\n{"tasks": [], "rationale": "ok"}\n```'
        assert clean_json_response(raw) == '{"tasks": [], "rationale": "ok"}'

    def test_clean_json_with_code_fence_no_lang(self):
        raw = '```\n{"tasks": [], "rationale": "ok"}\n```'
        assert clean_json_response(raw) == '{"tasks": [], "rationale": "ok"}'


class TestOrchestrator:
    def test_is_done_with_no_plan(self):
        orchestrator = Orchestrator(goal="test")
        assert orchestrator._is_done() is False

    def test_is_done_with_all_completed(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator.current_plan = Plan(
            tasks=[TaskModel(id="1", description="do thing")],
            rationale="ok",
        )
        orchestrator.current_plan.tasks[0].status = "completed"
        assert orchestrator._is_done() is True

    def test_is_done_with_failed(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator.current_plan = Plan(
            tasks=[TaskModel(id="1", description="do thing")],
            rationale="ok",
        )
        orchestrator.current_plan.tasks[0].status = "failed"
        assert orchestrator._is_done() is False

    def test_is_done_with_partial_completion(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator.current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a"),
                TaskModel(id="2", description="b"),
            ],
            rationale="ok",
        )
        orchestrator.current_plan.tasks[0].status = "completed"
        assert orchestrator._is_done() is False

    def test_is_done_with_empty_tasks(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator.current_plan = Plan(tasks=[], rationale="ok")
        assert orchestrator._is_done() is True

    def test_max_cycles_stops_loop(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator.cycles = 2
        orchestrator.settings.max_cycles = 2
        assert orchestrator.cycles >= orchestrator.settings.max_cycles
