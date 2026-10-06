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
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, Settings, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings_obj: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings_obj)
        self.settings = settings_obj or settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.all_tasks: list[Any] = []  # Track all tasks across cycles
        self.cycle_history: list[dict[str, Any]] = []  # History of each cycle

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            # Check max cycles limit
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[bold yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/bold yellow]")
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self.cycle_history)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.cycle_history.append({
                "cycle": self.cycles,
                "plan": plan.model_dump(),
                "tasks": [],
                "test_result": None,
            })
            return

        # Add new tasks to history
        for task in plan.tasks:
            task.cycle = self.cycles
            self.all_tasks.append(task)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        cycle_tasks = []
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
            cycle_tasks.append(task.model_dump())

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await self.tester.run(self.goal, plan.tasks)

        test_result_dict = test_result.model_dump()
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        self.cycle_history.append({
            "cycle": self.cycles,
            "plan": plan.model_dump(),
            "tasks": cycle_tasks,
            "test_result": test_result_dict,
        })

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False

        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")

        if failed > 0:
            return False
        if pending > 0:
            return False
        if completed >= len(self.all_tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks

    def print_summary(self) -> None:
        """Print a summary of all tasks and their status."""
        if not self.all_tasks:
            console.print("[yellow]No tasks executed.[/yellow]")
            return

        table = Table(title="Task Summary")
        table.add_column("Cycle", style="cyan")
        table.add_column("Task ID", style="magenta")
        table.add_column("Description", style="white")
        table.add_column("Status", style="green")
        table.add_column("Result", style="yellow", max_width=60)

        for task in self.all_tasks:
            status_color = {
                "completed": "green",
                "failed": "red",
                "pending": "yellow",
            }.get(task.status, "white")
            table.add_row(
                str(getattr(task, "cycle", "?")),
                task.id,
                task.description[:50],
                f"[{status_color}]{task.status}[/{status_color}]",
                (task.result or "")[:60],
            )

        console.print(table)
