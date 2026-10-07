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
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, output_callback: Callable[[str], Any] | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []
        self.output_callback = output_callback

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        if self.output_callback:
            await self._emit("Furrow started")
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            if self.output_callback:
                await self._emit(f"Cycle {self.cycles}")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                if self.output_callback:
                    await self._emit("Goal complete. Halting.")
                break
            if 0 < self.client.settings.max_cycles <= self.cycles:
                console.print(f"[yellow]Reached max_cycles ({self.client.settings.max_cycles}). Halting.[/yellow]")
                if self.output_callback:
                    await self._emit(f"Reached max_cycles ({self.client.settings.max_cycles}). Halting.")
                break

    async def _emit(self, message: str) -> None:
        if self.output_callback:
            try:
                result = self.output_callback(message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        if self.output_callback:
            await self._emit(f"Plan: {plan.rationale}")

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            if self.output_callback:
                await self._emit("No tasks planned.")
            return

        self.tasks = plan.tasks[: self.client.settings.max_parallel_tasks]

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in self.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(self.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                if self.output_callback:
                    await self._emit(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                if self.output_callback:
                    await self._emit(f"Task {task.id} completed: {result}")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            if self.output_callback:
                await self._emit(f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            if self.output_callback:
                await self._emit(f"Tests failed: {test_result.summary}")
            self.goal = (
                f"Fix failing tests:\nOriginal goal: {self.goal}\n"
                + "\n".join(test_result.failures)
            )

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
