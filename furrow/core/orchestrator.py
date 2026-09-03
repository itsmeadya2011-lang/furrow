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
from furrow.config import Plan, Settings, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.settings: Settings = self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"
            )
        )
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            should_continue = await self._cycle()
            if not should_continue:
                break
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(
                    f"[yellow]Reached max_cycles "
                    f"({self.settings.max_cycles}). Halting.[/yellow]"
                )
                break

    async def _cycle(self) -> bool:
        """Execute a single planning -> execution -> testing cycle.

        Returns False when the orchestrator should stop (e.g. planning
        failed), True otherwise.
        """
        try:
            with Status("[bold yellow]Planning...", console=console):
                plan = await self.planner.plan(self.goal)
        except Exception as e:
            console.print(
                Panel(
                    f"[red]Planning failed:[/red]\n{e}",
                    title=f"Cycle {self.cycles} — Planning Error",
                    border_style="red",
                )
            )
            return False

        self.plan = plan
        console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return True

        task_table = Table(title="Tasks", show_header=True, header_style="bold")
        task_table.add_column("ID", style="cyan", no_wrap=True)
        task_table.add_column("Description", style="white")
        task_table.add_column("Files", style="magenta")

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task in plan.tasks:
            task_table.add_row(
                task.id,
                task.description,
                ", ".join(task.files) if task.files else "-",
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
            else:
                task.status = "completed"
                task.result = result

        console.print(task_table)

        completed = sum(1 for t in plan.tasks if t.status == "completed")
        failed = sum(1 for t in plan.tasks if t.status == "failed")
        console.print(
            f"[bold]Results:[/bold] [green]{completed} completed[/green], "
            f"[red]{failed} failed[/red], {len(plan.tasks)} total"
        )
        for task in plan.tasks:
            style = "green" if task.status == "completed" else "red"
            console.print(f"  [{style}]• {task.id} -> {task.status}[/{style}]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        return True

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        if self.plan is not None:
            return self.plan.tasks
        return []
