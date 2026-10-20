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

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.client.settings.max_cycles if hasattr(self.client, "settings") else 0
        while True:
            if max_cycles and self.cycles >= max_cycles:
                console.print("[yellow]Reached max cycles. Halting.[/yellow]")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self.tasks = plan.tasks
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Execute tasks respecting dependencies in waves.
        pending = list(plan.tasks)
        completed_ids: set[str] = set()
        while pending:
            ready = [t for t in pending if all(d in completed_ids for d in (t.dependencies or []))]
            if not ready:
                # Deadlock: run remaining anyway.
                ready = pending[:]
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                workers = [WorkerAgent(task=task, client=self.client).run() for task in ready]
                results = await asyncio.gather(*workers, return_exceptions=True)
            for task, result in zip(ready, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    completed_ids.add(task.id)
                    console.print(f"[green]Task {task.id} completed[/green]")
            for task in ready:
                pending.remove(task)

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
        tasks = self._get_tasks()
        if not tasks:
            return True
        if any(t.status == "failed" for t in tasks):
            return False
        return all(t.status == "completed" for t in tasks)

    def _get_tasks(self) -> list[Any]:
        return self.tasks
