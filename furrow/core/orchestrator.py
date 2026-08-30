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
    def __init__(self, goal: str, client: LLMClient | None = None, on_event: Any = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self.on_event = on_event

    async def _emit(self, message: str) -> None:
        if self.on_event:
            if asyncio.iscoroutinefunction(self.on_event):
                await self.on_event(message)
            else:
                self.on_event(message)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                await self._emit(f"[yellow]Reached max cycles ({self.client.settings.max_cycles}). Halting.[/yellow]")
                console.print("[yellow]Reached max cycles. Halting.[/yellow]")
                break
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, workspace=self.client.settings.workspace)
        self.plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit(f"Planned {len(plan.tasks)} tasks")

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)
            async def run_with_limit(task):
                async with semaphore:
                    return await WorkerAgent(task=task, client=self.client).run()

            tasks = [run_with_limit(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
            await self._emit(f"Task {task.id} {task.status}")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks, workspace=self.client.settings.workspace)

        await self._emit(f"Tests {'passed' if test_result.passed else 'failed'}: {test_result.summary}")
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if self.plan is None:
            return False
        completed = sum(1 for t in self.plan.tasks if t.status == "completed")
        failed = sum(1 for t in self.plan.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.plan.tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.plan.tasks if self.plan else []
