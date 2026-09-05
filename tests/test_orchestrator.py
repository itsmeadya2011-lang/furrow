from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator, settings


class TestOrchestrator:
    def test_is_done_returns_true_when_all_completed(self):
        plan = Plan(
            tasks=[
                TaskModel(id="1", description="do thing", status="completed"),
                TaskModel(id="2", description="other thing", status="completed"),
            ],
            rationale="ok",
        )
        orchestrator = Orchestrator(goal="test")
        orchestrator._plan = plan
        assert orchestrator._is_done() is True

    def test_is_done_returns_false_when_any_failed(self):
        plan = Plan(
            tasks=[
                TaskModel(id="1", description="do thing", status="completed"),
                TaskModel(id="2", description="other thing", status="failed"),
            ],
            rationale="ok",
        )
        orchestrator = Orchestrator(goal="test")
        orchestrator._plan = plan
        assert orchestrator._is_done() is False

    def test_is_done_returns_true_when_no_tasks(self):
        orchestrator = Orchestrator(goal="test")
        assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_max_cycles_enforcement(self):
        original_max_cycles = settings.max_cycles
        settings.max_cycles = 2
        try:
            orchestrator = Orchestrator(goal="test")
            orchestrator._cycle = AsyncMock()  # type: ignore[method-assign]
            await orchestrator.run()
            assert orchestrator.cycles == 2
        finally:
            settings.max_cycles = original_max_cycles

    def test_goal_preserved_across_cycles(self):
        orchestrator = Orchestrator(goal="original goal")
        assert orchestrator.original_goal == "original goal"
