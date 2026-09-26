from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        output_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self.max_cycles: int = self.client.settings.max_cycles
        self.output_callback: Callable[[str], Awaitable[None]] | None = output_callback

    async def _emit(self, msg: str, style: str = "") -> None:
        """Emit a message to console and optionally to an output callback."""
        if self.output_callback:
            await self.output_callback(msg)
        if style:
            console.print(f"[{style}]{msg}[/{style}]")
        else:
            console.print(msg)

    async def run(self) -> None:
        logger.info("orchestrator_start", goal=self.goal, max_cycles=self.max_cycles)
        await self._emit(f"Furrow\nGoal: {self.goal}", style="bold green")

        while True:
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                logger.info("max_cycles_reached", cycles=self.cycles, max_cycles=self.max_cycles)
                await self._emit(
                    f"Reached max_cycles ({self.max_cycles}). Stopping.", style="yellow"
                )
                break
            self.cycles += 1
            await self._emit(f"\n\u2550\u2550\u2550 Cycle {self.cycles} \u2550\u2550\u2550", style="bold cyan")
            await self._cycle()
            if self._is_done():
                await self._emit("Goal complete. Halting.", style="bold green")
                logger.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        await self._emit(f"Plan rationale: {plan.rationale}", style="blue")
        logger.info("plan_created", tasks=len(plan.tasks))

        if not plan.tasks:
            await self._emit("No tasks planned. Goal may be complete.", style="yellow")
            return

        # Accumulate tasks across cycles for tracking
        self.all_tasks.extend(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"Task {task.id} failed: {result}", style="red")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"Task {task.id} completed", style="green")
                logger.info("task_completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=console):
            self.last_test_result = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )

        if self.last_test_result.passed:
            await self._emit(
                f"Tests passed: {self.last_test_result.summary}", style="green"
            )
            logger.info("tests_passed", summary=self.last_test_result.summary)
        else:
            await self._emit(f"Tests failed: {self.last_test_result.summary}", style="red")
            for failure in self.last_test_result.failures:
                await self._emit(f"  \u2022 {failure}", style="red")
            await self._emit("Will attempt fix in next cycle.", style="yellow")
            logger.warning("tests_failed", failures=self.last_test_result.failures)
            self.goal = f"Fix failing tests:\n" + "\n".join(self.last_test_result.failures)

    def _is_done(self) -> bool:
        """Check if the goal is complete."""
        tasks = self._get_tasks()
        if not tasks:
            return True
        if self.last_test_result is not None:
            return self.last_test_result.passed
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
