from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

import structlog

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult, settings as default_settings

log = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: Any | None = None,
        settings: Settings | None = None,
        on_event: Callable[[str], Any] | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or default_settings
        self.client = client
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.max_cycles = self.settings.max_cycles
        self.tasks: list[Any] = []
        self._on_event = on_event

    def _emit(self, message: str) -> None:
        """Emit a message to console (structlog) and optional event callback."""
        log.info("orchestrator", message=message)
        if self._on_event is not None:
            try:
                result = self._on_event(message)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception:
                log.exception("event_callback_failed")

    async def run(self) -> None:
        self._emit(f"Furrow starting\nGoal: {self.goal}")
        while True:
            self.cycles += 1
            self._emit(f"--- Cycle {self.cycles} ---")
            await self._cycle()
            if self._is_done():
                self._emit("Goal complete. Halting.")
                break
            if self.max_cycles and self.cycles >= self.max_cycles:
                self._emit(f"Reached max_cycles ({self.max_cycles}). Stopping.")
                break

    async def _cycle(self) -> None:
        self._emit("Planning...")
        plan: Plan = await self.planner.plan(self.goal)
        self._emit(f"Plan: {plan.rationale}\nTasks: {len(plan.tasks)}")

        # Store the plan's tasks so _is_done can inspect them
        self.tasks = plan.tasks

        if not self.tasks:
            self._emit("No tasks planned. Goal may be complete.")
            return

        self._emit(f"Executing {len(self.tasks)} tasks in parallel...")
        tasks = [
            WorkerAgent(task=task, client=self.client, settings=self.settings).run()
            for task in self.tasks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(self.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                self._emit(f"Task {task.id} completed")

        self._emit("Testing...")
        test_result: TestResult = await TesterAgent(
            client=self.client, settings=self.settings
        ).run(self.goal, self.tasks)

        if test_result.passed:
            self._emit(f"Tests passed: {test_result.summary}")
        else:
            self._emit(f"Tests failed: {test_result.summary}")
            for failure in test_result.failures:
                self._emit(f"  - {failure}")
            self._emit("Will attempt fix in next cycle.")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        return self.tasks