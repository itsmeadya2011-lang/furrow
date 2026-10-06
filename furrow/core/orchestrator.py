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
from furrow.config import Plan, Settings, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, output_callback: Any = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []
        self.output_callback = output_callback

    def _print(self, message: str, **kwargs: Any) -> None:
        if self.output_callback:
            self.output_callback(message)
        else:
            console.print(message, **kwargs)

    async def run(self) -> None:
        self._print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                self._print("[bold red]Max cycles reached. Halting.[/bold red]")
                break
            self.cycles += 1
            self._print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.tasks = plan.tasks

        max_parallel = self.client.settings.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_with_semaphore(task: Any) -> Any:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [run_with_semaphore(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        has_failed = False
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._print(f"[red]Task {task.id} failed: {result}[/red]")
                has_failed = True
            else:
                task.status = "completed"
                task.result = result
                self._print(f"[green]Task {task.id} completed[/green]")

        if has_failed:
            failed_tasks = [t for t in plan.tasks if t.status == "failed"]
            self._print(f"[yellow]Retrying {len(failed_tasks)} failed tasks once...[/yellow]")
            retry_tasks = failed_tasks
            retry_coroutines = [run_with_semaphore(t) for t in retry_tasks]
            retry_results = await asyncio.gather(*retry_coroutines, return_exceptions=True)
            for task, result in zip(retry_tasks, retry_results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    self._print(f"[red]Retry Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    self._print(f"[green]Retry Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self._print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._print(f"  • {failure}")
            self._print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return True
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.tasks
