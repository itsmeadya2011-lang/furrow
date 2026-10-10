from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult
from furrow.llm import LLMClient

settings = Settings()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []
        self.test_result: TestResult | None = None
        self.output_callback = output_callback
        self._console = Console()

    def _emit(self, message: str = "") -> None:
        """Send a message to both the rich console and the output callback."""
        self._console.print(message)
        if self.output_callback is not None:
            plain = self._console.export_text(message)
            self.output_callback(plain)

    async def run(self) -> None:
        self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self._console) as status:
            plan = await self.planner.plan(self.goal)
        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.tasks = []
            return

        self.tasks = list(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=self._console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._emit(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=self._console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self.test_result = test_result

        if test_result.passed:
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _get_tasks(self) -> list[Any]:
        return self.tasks

    def _is_done(self) -> bool:
        # Enforce max_cycles from settings (0 means infinite)
        if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
            return True

        tasks = self._get_tasks()

        # No tasks and goal complete
        if not tasks:
            return True

        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")

        # If any task failed and tests didn't pass, retry
        if failed > 0 and (self.test_result is None or not self.test_result.passed):
            return False

        # All tasks completed
        if completed >= len(tasks):
            return True

        return False