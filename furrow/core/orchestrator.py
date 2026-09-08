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
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, output_callback=None) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: dict[str, TaskModel] = {}
        self.completed_task_ids: set[str] = set()
        self.failed_task_ids: set[str] = set()
        self.output_callback = output_callback

    def _log(self, message: str, style: str = "") -> None:
        if self.output_callback:
            self.output_callback(message)
        else:
            console.print(message, style=style)

    async def run(self) -> None:
        self._log(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._log(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._log("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._log(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._log("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Register new tasks
        for task in plan.tasks:
            if task.id not in self.all_tasks:
                self.all_tasks[task.id] = task

        # Resolve task execution order based on dependencies
        executable_tasks = self._get_executable_tasks(plan.tasks)

        if not executable_tasks:
            self._log("[yellow]No executable tasks (all blocked by dependencies or completed).[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in executable_tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(executable_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.failed_task_ids.add(task.id)
                self._log(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.completed_task_ids.add(task.id)
                self._log(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, list(self.all_tasks.values()))

        if test_result.passed:
            self._log(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._log(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._log(f"  • {failure}")
            self._log("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Keep original goal but add test failures as context
            self.goal = f"{self.original_goal}\n\nPrevious test failures to fix:\n" + "\n".join(test_result.failures)

    def _get_executable_tasks(self, tasks: list[TaskModel]) -> list[TaskModel]:
        """Return tasks that have all dependencies completed and are not already completed."""
        executable = []
        for task in tasks:
            if task.id in self.completed_task_ids:
                continue
            if task.id in self.failed_task_ids:
                # Allow retry of failed tasks
                pass
            deps_met = all(dep_id in self.completed_task_ids for dep_id in task.dependencies)
            if deps_met:
                executable.append(task)
        return executable

    def _is_done(self) -> bool:
        # Check if all tasks in the current plan are completed
        if not self.all_tasks:
            return False

        pending = [t for t in self.all_tasks.values() if t.id not in self.completed_task_ids]
        if pending:
            return False

        # All tasks completed, check if we should continue
        return True

    def get_status(self) -> dict[str, Any]:
        return {
            "cycles": self.cycles,
            "original_goal": self.original_goal,
            "current_goal": self.goal,
            "total_tasks": len(self.all_tasks),
            "completed": len(self.completed_task_ids),
            "failed": len(self.failed_task_ids),
            "pending": len(self.all_tasks) - len(self.completed_task_ids) - len(self.failed_task_ids),
        }