from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, TaskModel, settings
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                console.print(
                    f"[bold yellow]Max cycles ({settings.max_cycles}) reached. Halting.[/bold yellow]"
                )
                break
            self.cycles += 1
            logger.info("cycle_started", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        logger.info("planning_complete", task_count=len(plan.tasks))
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge new tasks into persistent state, updating existing ones by id.
        for task in plan.tasks:
            existing = next((t for t in self._tasks if t.id == task.id), None)
            if existing is not None:
                # Preserve prior execution status (completed/failed) if already run.
                if existing.status in ("completed", "failed"):
                    task.status = existing.status
                    task.result = existing.result
            else:
                self._tasks.append(task)

        # Topologically sort so dependencies run first.
        ordered = self._topological_sort(self._tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            logger.info("tasks_executing", task_count=len(ordered))
            semaphore = asyncio.Semaphore(settings.max_parallel_tasks)

            async def _run_with_semaphore(task: TaskModel) -> tuple[TaskModel, Any]:
                async with semaphore:
                    result = await WorkerAgent(task=task, client=self.client).run()
                    return task, result

            wrapped = [_run_with_semaphore(task) for task in ordered]
            results = await asyncio.gather(*wrapped, return_exceptions=True)

        for item in results:
            if isinstance(item, Exception):
                console.print(f"[red]Worker error: {item}[/red]")
                continue
            task, result = item
            existing = next((t for t in self._tasks if t.id == task.id), None)
            if existing is not None:
                existing.status = "completed"
                existing.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
            else:
                task.status = "completed"
                task.result = result
                self._tasks.append(task)
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self._tasks)

        logger.info("testing_complete", passed=test_result.passed, summary=test_result.summary)
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _topological_sort(self, tasks: list[TaskModel]) -> list[TaskModel]:
        """Return tasks ordered so that dependencies run first.

        Tasks whose dependencies are all completed (or unknown) are
        eligible to run. Cycles are broken by ignoring back-edges.
        """
        task_map: dict[str, TaskModel] = {t.id: t for t in tasks}
        completed_ids = {t.id for t in tasks if t.status == "completed"}

        # Build adjacency list; a dependency edge means dependency must run first.
        # We process in rounds: eligible = no incomplete dependencies.
        remaining = list(tasks)
        ordered: list[TaskModel] = []
        while remaining:
            eligible = [
                t for t in remaining
                if all(
                    dep in completed_ids or dep not in task_map
                    for dep in t.dependencies
                )
            ]
            if not eligible:
                # Break cycles / unresolved deps by running remaining in original order.
                ordered.extend(remaining)
                break
            ordered.extend(eligible)
            for t in eligible:
                remaining.remove(t)
                completed_ids.add(t.id)
        return ordered

    def _is_done(self) -> bool:
        if not self._tasks:
            return False
        completed = sum(1 for t in self._tasks if t.status == "completed")
        failed = sum(1 for t in self._tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._tasks):
            return True
        return False