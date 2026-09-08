from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, on_output: Callable[[str], None] | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self.on_output: Callable[[str], None] | None = on_output

    async def run(self, on_output: Callable[[str], None] | None = None) -> None:
        if on_output is not None:
            self.on_output = on_output
        self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                self._emit(f"[yellow]Reached max_cycles ({settings.max_cycles}). Halting.[/yellow]")
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
        self._tasks = plan.tasks
        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        semaphore = asyncio.Semaphore(settings.max_parallel_tasks)

        async def _run_with_limit(task: TaskModel) -> Any:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [_run_with_limit(task) for task in plan.tasks]
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
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        # If no tasks were planned, treat as done (goal may already be complete).
        if completed >= len(self._get_tasks()):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks

    def _emit(self, text: Any) -> None:
        console.print(text)
        if self.on_output is not None:
            plain = re.sub(r'\[/?[^\]]*\]', '', str(text))
            self.on_output(plain)
