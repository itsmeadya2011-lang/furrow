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
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()

Callback = Callable[[str], Any] | None


class Orchestrator:
    def __init__(
        self, 
        goal: str, 
        client: LLMClient | None = None, 
        settings: Settings | None = None,
        on_output: Callback = None
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.on_output = on_output or (lambda msg: console.print(msg))

    def _emit(self, message: str) -> None:
        if self.on_output:
            self.on_output(message)

    async def run(self) -> None:
        self._emit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}")
        while True:
            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.cycles >= self.settings.max_cycles > 0:
                self._emit(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                break

    async def _cycle(self) -> None:
        self._emit("[bold yellow]Planning...[/bold yellow]")
        plan = await self.planner.plan(self.goal)
        self._emit(f"Plan: {plan.model_dump()}")

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self._merge_tasks(plan.tasks)

        pending_tasks = [t for t in self.tasks if t.status in ("pending", "failed")]

        if not pending_tasks:
            self._emit("[yellow]All tasks completed or failed.[/yellow]")
            return

        self._emit("[bold yellow]Executing tasks in parallel...[/bold yellow]")
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        
        async def run_with_semaphore(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client, settings=self.settings).run()
        
        tasks_to_run = [run_with_semaphore(task) for task in pending_tasks]
        results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

        for task, result in zip(pending_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._emit(f"[green]Task {task.id} completed: {result}[/green]")

        self._emit("[bold yellow]Testing...[/bold yellow]")
        test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, self.tasks)

        if test_result.passed:
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _merge_tasks(self, new_tasks: list[TaskModel]) -> None:
        existing_by_id = {t.id: t for t in self.tasks}
        for new_task in new_tasks:
            if new_task.id in existing_by_id:
                existing = existing_by_id[new_task.id]
                if existing.status in ("completed", "failed"):
                    existing.description = new_task.description
                    existing.files = new_task.files
                    existing.dependencies = new_task.dependencies
                else:
                    existing_by_id[new_task.id] = new_task
            else:
                existing_by_id[new_task.id] = new_task
        self.tasks = list(existing_by_id.values())

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
