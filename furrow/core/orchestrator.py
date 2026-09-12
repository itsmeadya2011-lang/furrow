from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, console: Console | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.consecutive_failures = 0
        self.all_tasks: dict[str, Any] = {}  # Persist tasks across cycles by ID
        self.last_test_passed: bool = False
        self.console = console or Console()

    async def run(self) -> None:
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            # Check max_cycles limit
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                self.console.print(f"[yellow]Reached max cycles ({self.client.settings.max_cycles}). Stopping.[/yellow]")
                break
            # Check consecutive failures
            if self.consecutive_failures >= self.client.settings.max_cycle_failures:
                self.console.print(f"[red]Reached max consecutive failures ({self.client.settings.max_cycle_failures}). Stopping.[/red]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console) as status:
            plan = await self.planner.plan(self.goal)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Persist tasks across cycles
        for task in plan.tasks:
            if task.id not in self.all_tasks:
                self.all_tasks[task.id] = task
            else:
                # Update the task description/files but keep status
                self.all_tasks[task.id].description = task.description
                self.all_tasks[task.id].files = task.files

        # Get the current tasks to execute (only pending ones)
        tasks_to_run = [t for t in plan.tasks if t.status == "pending"]
        if not tasks_to_run:
            self.console.print("[yellow]All tasks already attempted. Repanning...[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in tasks_to_run
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        cycle_had_failure = False
        for task, result in zip(tasks_to_run, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.all_tasks[task.id].status = "failed"
                self.all_tasks[task.id].result = str(result)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
                cycle_had_failure = True
            else:
                task.status = "completed"
                task.result = result
                self.all_tasks[task.id].status = "completed"
                self.all_tasks[task.id].result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=self.console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, list(self.all_tasks.values()))

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self.consecutive_failures = 0
            self.last_test_passed = True
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
            self.consecutive_failures += 1
            self.last_test_passed = False

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks.values() if t.status == "completed")
        failed = sum(1 for t in self.all_tasks.values() if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.all_tasks) and self.last_test_passed:
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return list(self.all_tasks.values())
