from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_passed: bool | None = None
        self._max_cycles = max_cycles or self.client.settings.max_cycles
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(
                self.client.settings.max_parallel_tasks
            )
        return self._semaphore

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        logger.info("orchestrator_start", goal=self.goal)
        while True:
            self.cycles += 1
            if self._max_cycles > 0 and self.cycles > self._max_cycles:
                console.print(
                    "[yellow]Max cycles reached. Halting.[/yellow]"
                )
                logger.info("max_cycles_reached", cycles=self.cycles)
                break
            console.print(
                f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]"
            )
            logger.info("cycle_start", cycle=self.cycles)
            await self._cycle()
            if self._is_done():
                console.print(
                    "[bold green]Goal complete. Halting.[/bold green]"
                )
                logger.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(
            Panel(
                Pretty(plan.model_dump()),
                title="Plan",
                border_style="blue",
            )
        )
        logger.info(
            "plan_created",
            tasks=len(plan.tasks),
            rationale=plan.rationale,
        )

        if not plan.tasks:
            console.print(
                "[yellow]No tasks planned. Goal may be complete.[/yellow]"
            )
            return

        # Track tasks for completion checking
        self.tasks = plan.tasks

        with Status(
            "[bold yellow]Executing tasks in parallel...",
            console=console,
        ):
            results = await asyncio.gather(
                *[self._run_task(task) for task in plan.tasks],
                return_exceptions=True,
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)

        with Status(
            "[bold yellow]Testing...", console=console
        ) as status:
            test_result = await TesterAgent(
                client=self.client
            ).run(self.goal, plan.tasks)

        self.last_test_passed = test_result.passed

        if test_result.passed:
            console.print(
                f"[green]Tests passed: {test_result.summary}[/green]"
            )
            logger.info("tests_passed", summary=test_result.summary)
        else:
            console.print(
                f"[red]Tests failed: {test_result.summary}[/red]"
            )
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print(
                "[yellow]Will attempt fix in next cycle.[/yellow]"
            )
            logger.warning(
                "tests_failed", failures=test_result.failures
            )
            self.goal = (
                "Fix failing tests:\n" + "\n".join(test_result.failures)
            )

    async def _run_task(self, task: TaskModel) -> str:
        """Execute a single task with concurrency limiting."""
        async with self.semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        if not self.tasks:
            return True
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        if completed < len(self.tasks):
            return False
        # All tasks completed — stay looping if tests failed so the
        # next cycle can attempt fixes.
        if self.last_test_passed is False:
            return False
        return True

    def _get_tasks(self) -> list[Any]:
        return self.tasks
