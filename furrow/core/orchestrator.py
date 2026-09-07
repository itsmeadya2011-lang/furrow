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
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Any = None

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("furrow_started", goal=self.goal)
        while True:
            self.cycles += 1
            if self.client.settings.max_cycles > 0 and self.cycles > self.client.settings.max_cycles:
                console.print(f"[yellow]Reached max_cycles ({self.client.settings.max_cycles}). Halting.[/yellow]")
                logger.info("furrow_max_cycles_reached", cycles=self.cycles, max_cycles=self.client.settings.max_cycles)
                break
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("furrow_cycle_started", cycle=self.cycles)
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("furrow_goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            self.plan = await self.planner.plan(self.goal)
        logger.info("furrow_plan_complete", tasks=len(self.plan.tasks), rationale=self.plan.rationale)
        console.print(Panel(Pretty(self.plan.model_dump()), title="Plan", border_style="blue"))

        if not self.plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in self.plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(self.plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                logger.error("task_failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                logger.info("task_completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.plan.tasks)

        if test_result.passed:
            logger.info("tests_passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            logger.info("furrow_done_no_tasks")
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            logger.info("furrow_not_done_failed_tasks", failed=failed, completed=completed, total=len(tasks))
            return False
        if completed >= len(tasks):
            logger.info("furrow_done_all_completed", completed=completed, total=len(tasks))
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        if self.plan is None:
            return []
        return self.plan.tasks
