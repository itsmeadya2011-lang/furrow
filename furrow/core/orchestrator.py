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
        while True:
            self.cycles += 1
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

        with Status("[bold yellow]Executing tasks with dependency ordering...", console=console):
            await self._execute_tasks(plan.tasks)

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
            return False  # No tasks planned yet, keep going
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        blocked = sum(1 for t in tasks if t.status == "blocked")
        if failed > 0:
            return False
        # Done when all tasks are completed (and none pending/blocked)
        return completed == len(tasks) and (blocked == 0)

    def _get_tasks(self) -> list[Any]:
        return self.tasks

    async def _execute_tasks(self, tasks: list[Any]) -> None:
        """Execute tasks respecting dependencies. Runs independent tasks in parallel."""
        task_map = {t.id: t for t in tasks}
        completed_ids: set[str] = set()
        failed_ids: set[str] = set()

        def is_ready(task: Any) -> bool:
            return all(
                dep in completed_ids or dep in failed_ids
                for dep in task.dependencies
                if dep in task_map
            )

        pending = [t for t in tasks if is_ready(t)]
        # Also include tasks with no deps in the workspace
        while pending:
            # Run ready tasks in parallel
            running = []
            for task in pending:
                worker = WorkerAgent(task=task, client=self.client)
                running.append((task, worker.run()))

            for task, coro in running:
                try:
                    result = await coro
                    task.status = "completed"
                    task.result = result
                    completed_ids.add(task.id)
                    console.print(f"[green]Task {task.id} completed[/green]")
                except Exception as e:
                    task.status = "failed"
                    task.result = str(e)
                    failed_ids.add(task.id)
                    console.print(f"[red]Task {task.id} failed: {e}[/red]")

            # Find newly ready tasks
            remaining = [t for t in tasks if t.status == "pending" and t.id not in completed_ids and t.id not in failed_ids]
            pending = [t for t in remaining if is_ready(t)]

            if not pending and remaining:
                # Deadlock: remaining tasks have deps on each other or on non-existent tasks
                console.print("[yellow]Warning: some tasks cannot run due to unmet dependencies.[/yellow]")
                for t in remaining:
                    t.status = "blocked"
                break
