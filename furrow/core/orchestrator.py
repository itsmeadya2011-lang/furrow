from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
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
from furrow.state import StateManager

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        resume: bool = False,
    ) -> None:
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.state = StateManager(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings, state=self.state)

        if resume and self.state.goal:
            console.print(f"[yellow]Resuming from previous state: {self.state.goal}[/yellow]")
            self.goal = self.state.goal
        else:
            self.goal = goal
            self.state.goal = goal
            if not self.state.original_goal:
                self.state.original_goal = goal
            self.state.max_cycles = self.settings.max_cycles

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))

        while True:
            if self.state.is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

            self.state.cycles += 1
            cycle = self.state.cycles
            console.print(f"\n[bold cyan]═══ Cycle {cycle} ═══[/bold cyan]")

            await self._cycle(cycle)

            if self.state.is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self, cycle: int) -> None:
        # Plan
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)

        console.print(Panel(Pretty(plan.model_dump()), title=f"Plan (Cycle {cycle})", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Store tasks in state
        self.state.tasks = [task.model_dump() for task in plan.tasks]

        # Execute tasks in parallel
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update task statuses
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                self.state.update_task(task.id, status="failed", result=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                self.state.update_task(task.id, status="completed", result=result)
                console.print(f"[green]Task {task.id} completed[/green]")

        # Test
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(
                self.goal, plan.tasks
            )

        # Record history
        self.state.add_history_entry(cycle, plan, test_result)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Update goal to include test failures for next cycle
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
            self.state.goal = self.goal

    def show_status(self) -> None:
        """Display current status."""
        table = Table(title="Furrow Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Goal", self.state.goal)
        table.add_row("Original Goal", self.state.original_goal)
        table.add_row("Cycles", str(self.state.cycles))
        table.add_row("Max Cycles", str(self.state.max_cycles) if self.state.max_cycles else "Unlimited")

        tasks = self.state.tasks
        if tasks:
            pending = sum(1 for t in tasks if t.get("status") == "pending")
            completed = sum(1 for t in tasks if t.get("status") == "completed")
            failed = sum(1 for t in tasks if t.get("status") == "failed")
            table.add_row("Tasks", f"{completed} completed, {pending} pending, {failed} failed")

        console.print(table)