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
        on_output: Callable[[str], Any] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._on_output = on_output

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.client.settings.max_cycles
        if max_cycles > 0 and self.cycles >= max_cycles:
            console.print(f"[yellow]Max cycles ({max_cycles}) reached. Halting.[/yellow]")
            return
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            if self._on_output:
                await self._on_output(f"Starting cycle {self.cycles}")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                if self._on_output:
                    await self._on_output("Goal complete")
                break
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print(f"[yellow]Max cycles ({max_cycles}) reached. Halting.[/yellow]")
                if self._on_output:
                    await self._on_output("Max cycles reached")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        if self._on_output:
            await self._on_output("Planning done")

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._tasks = []
            if self._on_output:
                await self._on_output("No tasks planned")
            return

        self._tasks = plan.tasks
        max_parallel = self.client.settings.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_task(task: TaskModel):
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks_coro = [run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks_coro, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                if self._on_output:
                    await self._on_output(f"Task {task.id} failed")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                if self._on_output:
                    await self._on_output(f"Task {task.id} completed")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            if self._on_output:
                await self._on_output("Tests passed")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
            if self._on_output:
                await self._on_output("Tests failed, will retry")

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            # No tasks in this cycle - check if any prior tasks failed
            # If no failures, goal is complete
            return True
        
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False  # Loop continues to fix failures
        
        completed = sum(1 for t in tasks if t.status == "completed")
        return completed == len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks
