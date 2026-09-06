from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or (client.settings if client else Settings())
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._console = console or Console()

    async def run(self) -> None:
        self._console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self._console.print(f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Stopping.[/yellow]")
                break
            self.cycles += 1
            self._console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self._console) as status:
            plan = await self.planner.plan(self.goal)
        self._console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Store tasks for state tracking across cycles
        self.tasks = plan.tasks

        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def run_task(task: TaskModel) -> Any:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=self._console):
            results = await asyncio.gather(
                *[run_task(task) for task in plan.tasks], return_exceptions=True
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=self._console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self._console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._console.print(f"  • {failure}")
            self._console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return True
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.tasks
