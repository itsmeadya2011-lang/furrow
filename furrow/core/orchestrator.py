from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TestResult, TaskModel, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        callback: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.callback = callback
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None

    def _emit(self, message: str) -> None:
        if self.callback:
            self.callback(message)

    async def run(self) -> None:
        self._emit(f"Furrow\nGoal: {self.goal}")
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._emit(f"\n═══ Cycle {self.cycles} ═══")
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                self._emit("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                console.print("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                break
            if self._is_done():
                self._emit("Goal complete. Halting.")
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        self._emit("Planning...")
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._emit(f"Plan: {plan.model_dump()}")
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.tasks = plan.tasks

        if not plan.tasks:
            self._emit("No tasks planned. Goal may be complete.")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self._emit("Executing tasks in parallel...")
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        failed_task_ids: list[str] = []
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                failed_task_ids.append(task.id)
                self._emit(f"Task {task.id} failed: {result}")
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._emit(f"Task {task.id} completed")
                console.print(f"[green]Task {task.id} completed[/green]")

        self._emit("Testing...")
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self.last_test_result = test_result

        if test_result.passed:
            self._emit(f"Tests passed: {test_result.summary}")
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit(f"Tests failed: {test_result.summary}")
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
                console.print(f"  • {failure}")
            retry_parts = list(test_result.failures)
            if failed_task_ids:
                retry_parts.append("Retry failed tasks: " + ", ".join(failed_task_ids))
            self.goal = "Fix failing tests:\n" + "\n".join(retry_parts)
            self._emit("Will attempt fix in next cycle.")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.tasks):
            if self.last_test_result is not None and not self.last_test_result.passed:
                return False
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
