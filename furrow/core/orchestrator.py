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
from furrow.config import Plan, TestResult, TaskModel, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self.max_cycles = settings.max_cycles

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                console.print(f"[yellow]Max cycles ({self.max_cycles}) reached. Stopping.[/yellow]")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
        self._print_summary()

    def _print_summary(self) -> None:
        console.print(f"\n[bold cyan]═══ Summary ═══[/bold cyan]")
        console.print(f"Total cycles: {self.cycles}")
        tasks = self._get_tasks()
        if tasks:
            completed = sum(1 for t in tasks if t.status == "completed")
            failed = sum(1 for t in tasks if t.status == "failed")
            pending = sum(1 for t in tasks if t.status == "pending")
            console.print(f"Tasks: {len(tasks)} total | {completed} completed | {failed} failed | {pending} pending")
            for t in tasks:
                status_color = "green" if t.status == "completed" else "red" if t.status == "failed" else "yellow"
                console.print(f"  [{status_color}]{t.id}[/{status_color}]: {t.description[:60]}... ({t.status})")
        else:
            console.print("No tasks recorded.")

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            if self._tasks and self._is_done():
                console.print("[green]All tasks completed. Goal achieved.[/green]")
            else:
                console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge new tasks into self._tasks, avoiding duplicates by id
        for new_task in plan.tasks:
            existing_idx = next((i for i, t in enumerate(self._tasks) if t.id == new_task.id), None)
            if existing_idx is not None:
                # Update existing task with new info, but preserve status if already completed/failed
                existing = self._tasks[existing_idx]
                if existing.status not in ("completed", "failed"):
                    self._tasks[existing_idx] = new_task
            else:
                self._tasks.append(new_task)

        # Get tasks to execute (only pending ones from self._tasks)
        tasks_to_run = [t for t in self._tasks if t.status == "pending"]

        if not tasks_to_run:
            console.print("[yellow]No pending tasks to execute.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in tasks_to_run
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        # Update task status in self._tasks
        for task, result in zip(tasks_to_run, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self._tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        for t in tasks:
            if t.status != "completed":
                return False
        return True

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks
