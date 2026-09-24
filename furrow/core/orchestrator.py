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
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, websocket: Any = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.websocket = websocket

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = settings.max_cycles
        while True:
            self.cycles += 1
            if max_cycles > 0 and self.cycles > max_cycles:
                console.print(f"[yellow]Max cycles ({max_cycles}) reached. Halting.[/yellow]")
                break
            await self._broadcast(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                await self._broadcast("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _broadcast(self, message: str) -> None:
        console.print(message)
        if self.websocket:
            try:
                await self.websocket.send_text(message + "\n")
            except Exception:
                pass

    async def _cycle(self) -> None:
        await self._broadcast("[bold yellow]Planning...[/bold yellow]")
        plan = await self.planner.plan(self.goal, self.all_tasks)
        await self._broadcast(f"[blue]Plan:[/blue] {plan.rationale}")
        for task in plan.tasks:
            await self._broadcast(f"  • Task {task.id}: {task.description} (files: {', '.join(task.files) if task.files else 'any'})")

        if not plan.tasks:
            await self._broadcast("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge new tasks with existing, preserving completed ones
        for new_task in plan.tasks:
            existing = next((t for t in self.all_tasks if t.id == new_task.id), None)
            if not existing:
                self.all_tasks.append(new_task)

        await self._broadcast("[bold yellow]Executing tasks in parallel...[/bold yellow]")
        pending_tasks = [t for t in plan.tasks if t.status == "pending"]
        if pending_tasks:
            worker_tasks = [
                WorkerAgent(task=task, client=self.client, workspace=settings.workspace).run()
                for task in pending_tasks
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

            for task, result in zip(pending_tasks, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    await self._broadcast(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    await self._broadcast(f"[green]Task {task.id} completed[/green]")

        await self._broadcast("[bold yellow]Testing...[/bold yellow]")
        test_result = await self.tester.run(self.goal, self.all_tasks)

        if test_result.passed:
            await self._broadcast(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._broadcast(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._broadcast(f"  • {failure}")
            await self._broadcast("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.all_tasks):
            return True
        return False
