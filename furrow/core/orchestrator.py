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
from furrow.config import Plan, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if settings.max_cycles and self.cycles >= settings.max_cycles:
                console.print("[yellow]Max cycles reached. Halting.[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.tasks = list(plan.tasks) if plan.tasks else self.tasks

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Execute tasks in dependency waves
        max_parallel = settings.max_parallel_tasks or len(plan.tasks)
        pending = {t.id: t for t in plan.tasks}
        completed_ids: set[str] = set()
        while pending:
            wave = [
                t for t in pending.values()
                if all(dep in completed_ids for dep in t.dependencies)
            ]
            if not wave:
                console.print("[red]Deadlock: unresolvable task dependencies.[/red]")
                for t in pending.values():
                    t.status = "failed"
                    t.result = "unresolvable dependency"
                break
            wave = wave[:max_parallel]
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                results = await asyncio.gather(
                    *[WorkerAgent(task=task, client=self.client).run() for task in wave],
                    return_exceptions=True,
                )
            for task, result in zip(wave, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    completed_ids.add(task.id)
                    console.print(f"[green]Task {task.id} completed[/green]")
                del pending[task.id]

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._get_tasks()):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.tasks
