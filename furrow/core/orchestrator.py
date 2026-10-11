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
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []
        self.max_cycles = self.client.settings.max_cycles
        self.max_parallel = self.client.settings.max_parallel_tasks
        self.last_test_passed: bool | None = None

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.max_cycles and self.cycles >= self.max_cycles:
                console.print(f"[bold yellow]Reached max cycles ({self.max_cycles}). Halting.[/bold yellow]")
                break

    async def _cycle(self) -> None:
        try:
            plan = await self.planner.plan(self.goal)
        except Exception as e:
            console.print(f"[red]Planning failed: {e}[/red]")
            return
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.last_test_passed = True
            return

        # Merge new tasks, ensuring unique IDs across cycles
        existing_ids = {t.id for t in self.tasks}
        for task in plan.tasks:
            if task.id in existing_ids:
                task.id = f"{task.id}-{self.cycles}"
            existing_ids.add(task.id)
            self.tasks.append(task)

        pending = [t for t in self.tasks if t.status == "pending"]
        if not pending:
            console.print("[yellow]All tasks already processed.[/yellow]")
            return

        # Respect max_parallel_tasks (0 or None = unlimited)
        to_run = pending[: self.max_parallel] if self.max_parallel else pending

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [WorkerAgent(task=task, client=self.client).run() for task in to_run]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(to_run, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        self.last_test_passed = test_result.passed
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return True
        if self.last_test_passed is False:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.tasks)