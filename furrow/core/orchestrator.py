from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, Settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self, goal: str, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tasks: list[Any] = []  # list of TaskModel
        self.cycles = 0

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                break
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Update self.tasks with new tasks (replace for simplicity)
        self.tasks = plan.tasks

        # Limit parallel tasks
        max_parallel = self.settings.max_parallel_tasks
        if max_parallel > 0:
            # Chunk tasks into groups of max_parallel
            for i in range(0, len(self.tasks), max_parallel):
                chunk = self.tasks[i : i + max_parallel]
                await self._execute_chunk(chunk)
        else:
            await self._execute_chunk(self.tasks)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(
                self.goal, self.tasks
            )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_chunk(self, tasks: list[Any]) -> None:
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coroutines = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in tasks
            ]
            results = await asyncio.gather(*coroutines, return_exceptions=True)

        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        failed = any(t.status == "failed" for t in self.tasks)
        if failed:
            return False
        completed = all(t.status == "completed" for t in self.tasks)
        return completed

    def _get_tasks(self) -> list[Any]:
        return self.tasks