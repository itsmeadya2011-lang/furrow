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
        max_cycles: int | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or (client.settings if client else Settings())
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = max_cycles if max_cycles is not None else self.settings.max_cycles
        self._current_plan: Plan | None = None
        self._last_test_result: TestResult | None = None

    async def run(self) -> None:
        logger.info("orchestrator_start", goal=self.goal, max_cycles=self.max_cycles)
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            self.cycles += 1
            if self.max_cycles > 0 and self.cycles > self.max_cycles:
                console.print("[yellow]Max cycles reached. Stopping.[/yellow]")
                logger.info("max_cycles_reached", cycles=self.cycles)
                break
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._current_plan = plan
        console.print(
            Panel(
                Pretty(plan.model_dump()),
                title="Plan",
                border_style="blue",
            )
        )

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned")
            return

        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def run_task(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status(
            "[bold yellow]Executing tasks in parallel...", console=console
        ) as status:
            coros = [run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

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

        with Status("[bold yellow]Testing...", console=console) as status:
            self._last_test_result = await TesterAgent(
                client=self.client
            ).run(self.goal, plan.tasks)

        test_result = self._last_test_result
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning(
                "tests_failed",
                summary=test_result.summary,
                failures=test_result.failures,
            )
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if self._last_test_result and not self._last_test_result.passed:
            return False
        all_completed = all(t.status == "completed" for t in tasks)
        if all_completed:
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        if self._current_plan is None:
            return []
        return self._current_plan.tasks
