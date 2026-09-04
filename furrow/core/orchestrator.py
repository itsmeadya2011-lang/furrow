from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, settings as default_settings, Settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        config: Settings | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.last_plan: Plan | None = None
        self.config = config or default_settings
        self.on_progress = on_progress

    async def _progress(self, message: str) -> None:
        if self.on_progress is not None:
            await self.on_progress(message)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.config.max_cycles
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._progress(f"Cycle {self.cycles} starting")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print(f"[yellow]Max cycles ({max_cycles}) reached. Halting.[/yellow]")
                break

    async def _cycle(self) -> None:
        await self._progress("Planning...")
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        await self._progress(f"Plan received: {len(plan.tasks)} tasks")
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self.last_plan = plan

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        await self._progress("Executing tasks...")
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run(workspace=self.config.workspace)
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._progress(f"Task {task.id} failed")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._progress(f"Task {task.id} completed")

        await self._progress("Testing...")
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            await self._progress(f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            await self._progress(f"Tests failed: {test_result.summary}")
            self.goal = (
                f"{self.goal}\n\nFix failing tests:\n"
                + "\n".join(test_result.failures)
            )

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0 and completed == 0:
            return True
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        return self.last_plan.tasks if self.last_plan else []
