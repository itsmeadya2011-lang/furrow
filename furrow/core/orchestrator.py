from __future__ import annotations

import asyncio
import json
import logging
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
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()
logger = logging.getLogger("furrow")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        from furrow.config import settings as default_settings

        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                console.print(
                    f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]"
                )
                break

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
            self.all_tasks.extend(plan.tasks)
            return

        # Merge new tasks into the tracked task list (avoid duplicates by id)
        existing_ids = {t.id for t in self.all_tasks}
        for task in plan.tasks:
            if task.id not in existing_ids:
                self.all_tasks.append(task)
                existing_ids.add(task.id)

        # Execute tasks respecting dependencies
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
            self.goal = (
                "Fix failing tests:\n" + "\n".join(test_result.failures)
            )

    async def _execute_tasks(self, tasks: list[TaskModel]) -> None:
        """Execute tasks, respecting dependencies. Tasks with no unmet
        dependencies run in parallel; dependent tasks wait for their deps."""
        pending = {t.id: t for t in tasks}
        completed_ids: set[str] = set()

        while pending:
            # Find tasks whose dependencies are all completed
            ready = [
                t for t in pending.values()
                if all(dep in completed_ids for dep in t.dependencies)
            ]
            if not ready:
                # Circular dependency or missing dependency — run remaining anyway
                console.print(
                    "[yellow]Warning: circular or missing dependency detected. "
                    "Running remaining tasks.[/yellow]"
                )
                ready = list(pending.values())

            results = await asyncio.gather(
                *(WorkerAgent(task=t, client=self.client).run() for t in ready),
                return_exceptions=True,
            )

            for task, result in zip(ready, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    logger.error("Task %s failed: %s", task.id, result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    logger.info("Task %s completed", task.id)
                    console.print(f"[green]Task {task.id} completed[/green]")

                completed_ids.add(task.id)
                del pending[task.id]

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks
