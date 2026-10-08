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

default_console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        console: Console | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient()
        self.console = console if console is not None else default_console
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []

    async def run(self) -> None:
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            max_cycles = self.client.settings.max_cycles
            if max_cycles > 0 and self.cycles > max_cycles:
                self.console.print(f"[bold yellow]Reached max_cycles limit ({max_cycles}). Halting.[/bold yellow]")
                break

            await self._cycle()
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console) as status:
            plan = await self.planner.plan(self.goal)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Store the planned tasks so _get_tasks() can return them
        self.tasks = plan.tasks

        # Execute tasks respecting dependencies and max_parallel_tasks
        await self._execute_tasks(plan.tasks)

        with Status("[bold yellow]Testing...", console=self.console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Preserve the original goal, append fix context
            fix_context = "\n".join(test_result.failures)
            self.goal = f"{self.original_goal}\n\nFix failing tests:\n{fix_context}"

    async def _execute_tasks(self, tasks: list[Any]) -> None:
        """Execute tasks respecting dependencies and max_parallel_tasks limit."""
        max_parallel = self.client.settings.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel)

        completed_ids: set[str] = set()
        pending = list(tasks)

        while pending:
            # Find tasks whose dependencies are all completed
            ready = [t for t in pending if all(d in completed_ids for d in t.dependencies)]
            if not ready:
                # Deadlock: remaining tasks depend on each other or on missing IDs
                self.console.print(
                    f"[red]Warning: {len(pending)} task(s) cannot run due to unmet dependencies. "
                    f"Remaining: {[t.id for t in pending]}[/red]"
                )
                for t in pending:
                    t.status = "failed"
                    t.result = "Unmet dependencies"
                break

            # Run ready tasks in bounded parallelism
            async def run_one(task: Any) -> None:
                async with semaphore:
                    worker = WorkerAgent(task=task, client=self.client)
                    try:
                        result = await worker.run()
                        task.status = "completed"
                        task.result = result
                        self.console.print(f"[green]Task {task.id} completed[/green]")
                    except Exception as exc:
                        task.status = "failed"
                        task.result = str(exc)
                        self.console.print(f"[red]Task {task.id} failed: {exc}[/red]")
                    finally:
                        completed_ids.add(task.id)

            await asyncio.gather(*(run_one(t) for t in ready))
            # Remove completed tasks from pending
            pending = [t for t in pending if t.id not in completed_ids]

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True

        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")

        # If any task failed, retry the cycle
        if failed > 0:
            return False

        # All tasks completed
        if completed >= len(tasks):
            return True

        # Some tasks still pending/in_progress
        return False

    def _get_tasks(self) -> list[Any]:
        return self.tasks