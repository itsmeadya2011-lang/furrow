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
    def __init__(self, goal: str, client: LLMClient | None = None, message_queue: asyncio.Queue | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.settings = settings
        self.tasks: list[Any] = []
        self.message_queue = message_queue

    def _emit(self, message: str) -> None:
        if self.message_queue is not None:
            self.message_queue.put_nowait(message)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        self._emit(f"Goal: {self.goal}")
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit(f"═══ Cycle {self.cycles} ═══")
            await self._cycle()
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print("[yellow]Reached max_cycles limit.[/yellow]")
                self._emit("Reached max_cycles limit.")
                break
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._emit("Goal complete. Halting.")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit(f"Plan: {len(plan.tasks)} tasks")

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._emit("No tasks planned.")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                self._emit(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                self._emit(f"Task {task.id} completed")

        self.tasks.extend(plan.tasks)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._emit(f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            self._emit(f"Tests failed: {test_result.summary}")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
                self._emit(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self._emit("Will attempt fix in next cycle.")
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
