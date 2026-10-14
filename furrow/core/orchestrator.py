from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

log = get_logger(__name__)


class Orchestrator:
    MAX_CONSECUTIVE_ERRORS: int = 3

    def __init__(self, goal: str, client: LLMClient | None = None, console: Console | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._consecutive_errors = 0
        self.console = console or Console()

    async def run(self) -> None:
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        log.info("orchestrator.start", goal=self.goal)
        while True:
            self.cycles += 1
            log.info("cycle.start", cycle=self.cycles)
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            try:
                await self._cycle()
            except Exception as e:
                self._consecutive_errors += 1
                log.error(
                    "cycle.error",
                    cycle=self.cycles,
                    error=str(e),
                    consecutive_errors=self._consecutive_errors,
                )
                self.console.print(f"[red]Cycle {self.cycles} failed: {e}[/red]")
                if self._consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    self.console.print(
                        f"[bold red]{self.MAX_CONSECUTIVE_ERRORS} consecutive errors. "
                        "Stopping orchestrator.[/bold red]"
                    )
                    break
                continue
            self._consecutive_errors = 0
            if self._is_done():
                log.info("orchestrator.complete", cycles=self.cycles)
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        log.info("planning.start")
        with Status("[bold yellow]Planning...", console=self.console) as status:
            plan = await self.planner.plan(self.goal)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        log.info("planning.complete", task_count=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            log.warning("planning.no_tasks")
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to tracked tasks (only if they don't already exist by ID)
        for task in plan.tasks:
            if not any(existing.id == task.id for existing in self._tasks):
                self._tasks.append(task)

        log.info("execution.start", task_ids=[t.id for t in plan.tasks])
        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            # Update task status in tracked tasks
            for tracked in self._tasks:
                if tracked.id == task.id:
                    if isinstance(result, Exception):
                        tracked.status = "failed"
                        tracked.result = str(result)
                        log.error("task.failed", task_id=task.id, error=str(result))
                        self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    else:
                        tracked.status = "completed"
                        tracked.result = result
                        log.info("task.completed", task_id=task.id)
                        self.console.print(f"[green]Task {task.id} completed[/green]")
                    break

        log.info("testing.start")
        with Status("[bold yellow]Testing...", console=self.console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            log.info("testing.passed", summary=test_result.summary)
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            log.warning("testing.failed", summary=test_result.summary, failures=test_result.failures)
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            log.info("testing.retry_scheduled")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if len(tasks) == 0:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks