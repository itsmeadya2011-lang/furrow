"""Tests for orchestrator message_sink behavior and web server endpoint registration.

No real LLM calls are made — planner, worker, and tester are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class TestOrchestratorMessageSink:
    """Verify the message_sink receives plain-text status updates."""

    async def test_message_sink_receives_messages(self) -> None:
        orch = Orchestrator.__new__(Orchestrator)
        orch.goal = "test goal"
        orch.settings = MagicMock(max_cycles=0)
        orch.client = MagicMock()
        orch.planner = MagicMock()
        orch.cycles = 0
        orch.tasks = []
        orch.test_results = []
        orch.last_plan = None

        received: list[str] = []

        async def sink(msg: str) -> None:
            received.append(msg)

        orch.message_sink = sink

        # Mock the plan to have no tasks (terminates immediately)
        orch.planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="no tasks"))

        await orch.run()

        # The sink should have received at least the start message and cycle message
        assert len(received) >= 2
        assert any("Furrow started" in m for m in received)
        assert any("Cycle 1" in m for m in received)
        assert any("Goal complete" in m for m in received)

    async def test_message_sink_not_required(self) -> None:
        """Orchestrator should work fine without a message_sink (CLI usage)."""
        orch = Orchestrator.__new__(Orchestrator)
        orch.goal = "test goal"
        orch.settings = MagicMock(max_cycles=0)
        orch.client = MagicMock()
        orch.planner = MagicMock()
        orch.planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="done"))
        orch.cycles = 0
        orch.tasks = []
        orch.test_results = []
        orch.last_plan = None
        orch.message_sink = None

        # Should not raise
        await orch.run()

    async def test_message_sink_isolation_from_console_errors(self) -> None:
        """If message_sink raises, the orchestrator should not crash."""
        orch = Orchestrator.__new__(Orchestrator)
        orch.goal = "test goal"
        orch.settings = MagicMock(max_cycles=0)
        orch.client = MagicMock()
        orch.planner = MagicMock()
        orch.planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="done"))
        orch.cycles = 0
        orch.tasks = []
        orch.test_results = []
        orch.last_plan = None

        async def broken_sink(msg: str) -> None:
            raise RuntimeError("WebSocket closed")

        orch.message_sink = broken_sink

        # Should not raise despite broken sink
        await orch.run()


class TestOrchestratorCycle:
    """Verify a full cycle with mocked collaborators."""

    async def test_full_cycle_with_task_completion(self) -> None:
        from furrow.agents.tester import TesterAgent
        from furrow.agents.worker import WorkerAgent

        orch = Orchestrator.__new__(Orchestrator)
        orch.goal = "add a feature"
        orch.settings = MagicMock(max_cycles=0)
        orch.cycles = 0
        orch.tasks = []
        orch.test_results = []
        orch.last_plan = None
        orch.message_sink = None

        plan = Plan(
            tasks=[TaskModel(id="1", description="write code", files=["src/main.py"])],
            rationale="one task",
        )
        orch.planner = MagicMock()
        orch.planner.plan = AsyncMock(return_value=plan)

        orch.client = MagicMock()

        # Mock worker results
        with patch.object(WorkerAgent, "run", new_callable=AsyncMock) as mock_worker:
            mock_worker.return_value = "Implemented the feature"

            # Mock tester
            with patch.object(TesterAgent, "run", new_callable=AsyncMock) as mock_tester:
                mock_tester.return_value = TestResult(
                    passed=True, summary="All tests passed", failures=[]
                )

                await orch._cycle()

                # Tasks should be completed
                assert orch.tasks[0].status == "completed"
                assert orch.tasks[0].result == "Implemented the feature"
                # Test result should be recorded
                assert len(orch.test_results) == 1
                assert orch.test_results[0].passed is True

    async def test_cycle_resets_goal_on_test_failure(self) -> None:
        from furrow.agents.tester import TesterAgent
        from furrow.agents.worker import WorkerAgent

        orch = Orchestrator.__new__(Orchestrator)
        orch.goal = "original goal"
        orch.settings = MagicMock(max_cycles=0)
        orch.cycles = 0
        orch.tasks = []
        orch.test_results = []
        orch.last_plan = None
        orch.message_sink = None

        plan = Plan(
            tasks=[TaskModel(id="1", description="fix bug", files=["src/bug.py"])],
            rationale="one task",
        )
        orch.planner = MagicMock()
        orch.planner.plan = AsyncMock(return_value=plan)
        orch.client = MagicMock()

        with patch.object(WorkerAgent, "run", new_callable=AsyncMock) as mock_worker:
            mock_worker.return_value = "fixed"

            with patch.object(TesterAgent, "run", new_callable=AsyncMock) as mock_tester:
                mock_tester.return_value = TestResult(
                    passed=False,
                    summary="tests failed",
                    failures=["test_foo failed"],
                )

                await orch._cycle()

                # Goal should be updated to include the fix context
                assert "Fix failing tests" in orch.goal
                assert "test_foo failed" in orch.goal


class TestWebSocketServer:
    """Verify the FastAPI app has the expected routes registered."""

    def test_index_route_exists(self) -> None:
        from furrow.web.server import app

        routes = {r.path for r in app.routes}
        assert "/" in routes

    def test_websocket_route_exists(self) -> None:
        from furrow.web.server import app

        routes = {r.path for r in app.routes}
        assert "/ws" in routes

    def test_start_request_model(self) -> None:
        from furrow.web.server import StartRequest

        req = StartRequest(goal="test")
        assert req.goal == "test"
        assert req.model is None

    def test_start_request_with_model(self) -> None:
        from furrow.web.server import StartRequest

        req = StartRequest(goal="test", model="gpt-4")
        assert req.goal == "test"
        assert req.model == "gpt-4"
