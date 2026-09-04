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

_default_console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, console: Console | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self.console = console or _default_console

    async def run(self) -> None:
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console) as status:
            self.plan = await self.planner.plan(self.goal)
        self.console.print(Panel(Pretty(self.plan.model_dump()), title="Plan", border_style="blue"))

        if not self.plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in self.plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(self.plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=self.console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.plan.tasks)

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.plan.tasks if self.plan else []
