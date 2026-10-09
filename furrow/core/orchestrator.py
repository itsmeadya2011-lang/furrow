from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_message: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = self.client.settings.max_cycles
        self.all_tasks: dict[str, TaskModel] = {}
        self.last_test_result: TestResult | None = None
        self._on_message = on_message

    def _log(self, message: str) -> None:
        console.print(message)
        if self._on_message:
            self._on_message(message)

    async def run(self) -> None:
        self._log(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._log(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._log("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                self._log(f"[bold yellow]Max cycles ({self.max_cycles}) reached. Halting.[/bold yellow]")
                break

    async def _cycle(self) -> None:
        failed_tasks = [t for t in self.all_tasks.values() if t.status == "failed"]
        goal_context = self.goal
        if failed_tasks:
            failed_info = "\n".join(f"Task {t.id} failed: {t.result}" for t in failed_tasks)
            goal_context = f"{self.goal}\n\nPreviously failed tasks:\n{failed_info}"

        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(goal_context)
        self._log(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._log("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        for task in plan.tasks:
            self.all_tasks[task.id] = task

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
                self._log(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._log(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self.last_test_result = test_result

        if test_result.passed:
            self._log(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._log(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._log(f"  • {failure}")
            self._log("[yellow]Will attempt fix in next cycle.[/yellow]")

    def _is_done(self) -> bool:
        if self.last_test_result and not self.last_test_result.passed:
            return False

        if self.last_test_result and self.last_test_result.passed:
            return True

        tasks = self._get_tasks()
        if not tasks:
            return False

        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return list(self.all_tasks.values())
