from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult, settings as default_settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        # Tasks tracked across cycles so completion can be evaluated correctly.
        self._tasks: dict[str, Any] = {}

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.settings.max_cycles and self.cycles >= self.settings.max_cycles:
                console.print(
                    f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Halting.[/yellow]"
                )
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            if not self._tasks:
                return
            # No new tasks but nothing pending either -> done.
            return

        # Register new tasks (by id) so they persist across cycles.
        for task in plan.tasks:
            if task.id not in self._tasks:
                self._tasks[task.id] = task

        pending = [t for t in self._tasks.values() if t.status == "pending"]
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *(WorkerAgent(task=task, client=self.client).run() for task in pending),
                return_exceptions=True,
            )

        for task, result in zip(pending, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, list(self._tasks.values()))

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            # Reframe the goal to focus on failures and let the planner re-slice.
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return all(t.status == "completed" for t in tasks)

    def _get_tasks(self) -> list[Any]:
        return list(self._tasks.values())
