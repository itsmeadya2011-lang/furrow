from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        state_file: str | Path | None = None,
        output_callback=None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.state_file = Path(state_file) if state_file else Path(self.settings.workspace) / ".furrow_state.json"
        self.output_callback = output_callback
        self._load_state()

    def _print(self, message: str, style: str = "") -> None:
        """Print to console and callback if provided."""
        if style:
            console.print(f"[{style}]{message}[/{style}]")
        else:
            console.print(message)
        if self.output_callback:
            self.output_callback(message)

    def _load_state(self) -> None:
        """Load previous state from file."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.cycles = data.get("cycles", 0)
                self.all_tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                self.goal = data.get("goal", self.goal)
                self.original_goal = data.get("original_goal", self.original_goal)
                self._print(f"[dim]Loaded state from {self.state_file} (cycle {self.cycles})[/dim]")
            except Exception as e:
                self._print(f"[yellow]Failed to load state: {e}[/yellow]")

    def _save_state(self) -> None:
        """Save current state to file."""
        data = {
            "cycles": self.cycles,
            "goal": self.goal,
            "original_goal": self.original_goal,
            "tasks": [t.model_dump() for t in self.all_tasks],
            "timestamp": time.time(),
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    async def run(self) -> None:
        self._print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.settings.max_cycles

        while True:
            if max_cycles > 0 and self.cycles >= max_cycles:
                self._print(f"[yellow]Reached max cycles ({max_cycles}). Stopping.[/yellow]")
                break

            self.cycles += 1
            self._print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()

            if self._is_done():
                self._print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self.all_tasks)
        self._print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to our tracking list
        for task in plan.tasks:
            # Check if task already exists (by id)
            existing = next((t for t in self.all_tasks if t.id == task.id), None)
            if existing:
                # Update existing task
                existing.description = task.description
                existing.files = task.files
                existing.dependencies = task.dependencies
                existing.status = "pending"
                existing.result = None
            else:
                self.all_tasks.append(task)

        # Filter tasks that need to run (pending or failed)
        # Also respect dependencies - only run tasks whose dependencies are completed
        tasks_to_run = []
        for task in plan.tasks:
            if task.status not in ("pending", "failed"):
                continue

            # Check if all dependencies are completed
            deps_met = True
            for dep_id in task.dependencies:
                dep_task = next((t for t in self.all_tasks if t.id == dep_id), None)
                if not dep_task or dep_task.status != "completed":
                    deps_met = False
                    break

            if deps_met:
                tasks_to_run.append(task)
            else:
                self._print(f"[yellow]Task {task.id} waiting for dependencies: {task.dependencies}[/yellow]")

        if not tasks_to_run:
            self._print("[yellow]No tasks ready to execute this cycle (waiting on dependencies).[/yellow]")
            return

        # Respect max_parallel_tasks
        max_parallel = self.settings.max_parallel_tasks
        if len(tasks_to_run) > max_parallel:
            self._print(f"[yellow]Limiting to {max_parallel} parallel tasks (of {len(tasks_to_run)})[/yellow]")
            tasks_to_run = tasks_to_run[:max_parallel]

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in tasks_to_run
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(tasks_to_run, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        if test_result.passed:
            self._print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._print(f"  • {failure}")
            self._print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Keep original goal but add fix context
            self.goal = f"{self.original_goal}\n\nPrevious test failures to fix:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        # Get all tasks from the latest plan (which should be all tracked tasks)
        pending = [t for t in self.all_tasks if t.status == "pending"]
        failed = [t for t in self.all_tasks if t.status == "failed"]
        completed = [t for t in self.all_tasks if t.status == "completed"]

        if failed:
            return False
        if not pending and not failed and completed:
            return True
        if not self.all_tasks:
            return False
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks

    def print_status(self) -> None:
        """Print a summary table of all tasks."""
        table = Table(title="Task Status")
        table.add_column("ID", style="cyan")
        table.add_column("Description")
        table.add_column("Files")
        table.add_column("Status", style="bold")
        table.add_column("Result")

        for task in self.all_tasks:
            status_style = {
                "pending": "yellow",
                "completed": "green",
                "failed": "red",
            }.get(task.status, "white")
            table.add_row(
                task.id,
                task.description,
                ", ".join(task.files) if task.files else "any",
                f"[{status_style}]{task.status}[/{status_style}]",
                (task.result[:50] + "...") if task.result and len(task.result) > 50 else (task.result or ""),
            )
        console.print(table)
