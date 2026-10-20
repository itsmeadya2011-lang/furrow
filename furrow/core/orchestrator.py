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
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []
        self._done: bool = False

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.client.settings.max_cycles or 0
        while True:
            self.cycles += 1
            if max_cycles and self.cycles > max_cycles:
                console.print("[yellow]Reached max cycles. Halting.[/yellow]")
                break
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
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

        self.tasks = plan.tasks

        # Execute in dependency-aware waves
        remaining = list(plan.tasks)
        completed_ids: set[str] = set()
        while remaining:
            wave = [t for t in remaining if all(d in completed_ids for d in t.dependencies)]
            if not wave:
                # deadlock; run all remaining
                wave = remaining
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
            for t in wave:
                remaining = [r for r in remaining if r.id != t.id]

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._done = True
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if any(t.status == "failed" for t in self._get_tasks()):
            return False
        return self._done and len(self._get_tasks()) > 0

    def _get_tasks(self) -> list[Any]:
        return self.tasks
