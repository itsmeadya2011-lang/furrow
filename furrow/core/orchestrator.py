from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int | None = None,
        on_output: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.current_tasks: list[Any] = []
        self.tests_passed: bool = False
        self._settings = Settings()
        self.max_cycles = max_cycles if max_cycles is not None else self._settings.max_cycles
        self.on_output = on_output

    def _emit(self, message: str) -> None:
        if self.on_output is not None:
            self.on_output(message)
        else:
            console.print(message)

    async def run(self, on_output: Optional[Callable[[str], None]] = None) -> None:
        if on_output is not None:
            self.on_output = on_output

        self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))

        while True:
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                self._emit(f"[bold yellow]Max cycles ({self.max_cycles}) reached. Halting.[/bold yellow]")
                break

            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.current_tasks = []
            self.tests_passed = False
            return

        self.current_tasks = plan.tasks

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
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

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self.tests_passed = True
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.tests_passed = False
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.current_tasks:
            return False

        completed = sum(1 for t in self.current_tasks if t.status == "completed")
        failed = sum(1 for t in self.current_tasks if t.status == "failed")

        if failed > 0:
            return False

        if completed >= len(self.current_tasks) and self.tests_passed:
            return True

        return False

    def _get_tasks(self) -> list[Any]:
        return self.current_tasks