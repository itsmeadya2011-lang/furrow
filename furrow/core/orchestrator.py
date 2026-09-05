from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
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

settings = Settings()
console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_output: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._plan: Plan | None = None
        self._on_output = on_output

    def _output(self, text: str) -> None:
        console.print(text)
        if self._on_output is not None:
            self._on_output(text)

    async def run(self) -> None:
        self._output(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        if settings.max_cycles > 0:
            self._output(f"[dim]Max cycles: {settings.max_cycles}[/dim]")
        while True:
            self.cycles += 1
            self._output(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                self._output("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                break
            await self._cycle()
            if self._is_done():
                self._output("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._plan = plan
        self._output(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._output("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

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
                self._output(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._output(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self._output(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._output(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._output(f"  • {failure}")
            self._output("[yellow]Will attempt fix in next cycle.[/yellow]")
            fix_context = "Fix failing tests:\n" + "\n".join(test_result.failures)
            self.goal = f"{self.original_goal}\n\nContext from previous cycle:\n{fix_context}"

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
        if self._plan is not None:
            return self._plan.tasks
        return []
