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
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.completed_task_ids: set[str] = set()
        self.failed_task_ids: set[str] = set()

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Stopping.[/yellow]")
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        # Filter out already completed tasks
        pending_tasks = [t for t in self.all_tasks if t.id not in self.completed_task_ids and t.id not in self.failed_task_ids]

        if not pending_tasks:
            with Status("[bold yellow]Planning...", console=console) as status:
                plan = await self.planner.plan(self.goal, self._get_context())
            console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

            if not plan.tasks:
                console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
                return

            # Add new tasks to our tracking
            for task in plan.tasks:
                if task.id not in [t.id for t in self.all_tasks]:
                    self.all_tasks.append(task)
            pending_tasks = plan.tasks

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in pending_tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(pending_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.failed_task_ids.add(task.id)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.completed_task_ids.add(task.id)
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, self.all_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if self.failed_task_ids:
            return False
        pending = [t for t in self.all_tasks if t.id not in self.completed_task_ids]
        if not pending and self.all_tasks:
            return True
        return False

    def _get_context(self) -> str:
        """Build context string for the planner including existing files and task history."""
        context_parts = []

        # Add workspace file listing
        files = self.client.list_files(self.settings.workspace)
        if files:
            context_parts.append(f"Existing files in workspace:\n" + "\n".join(f"  - {f}" for f in files[:50]))

        # Add completed task summaries
        completed = [t for t in self.all_tasks if t.id in self.completed_task_ids]
        if completed:
            context_parts.append("Completed tasks:")
            for t in completed:
                context_parts.append(f"  - {t.id}: {t.description} -> {t.result[:200] if t.result else 'done'}")

        # Add failed task info
        failed = [t for t in self.all_tasks if t.id in self.failed_task_ids]
        if failed:
            context_parts.append("Failed tasks (need fixing):")
            for t in failed:
                context_parts.append(f"  - {t.id}: {t.description} -> ERROR: {t.result[:200] if t.result else 'unknown'}")

        return "\n\n".join(context_parts) if context_parts else "No previous context."

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks