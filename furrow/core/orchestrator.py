from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Callable

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent, WorkerResult
from furrow.config import Plan, Settings, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger("furrow.orchestrator")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_output: Callable[[str], Any] | None = None,
    ) -> None:
        self.goal = goal
        self.cfg = settings or default_settings
        self.client = client or LLMClient(settings=self.cfg)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []
        self.on_output = on_output

    def _log(self, message: str) -> None:
        """Emit a message to both the console and an optional output callback."""
        console.print(message)
        if self.on_output is not None:
            try:
                if asyncio.iscoroutinefunction(self.on_output):
                    asyncio.create_task(self.on_output(message))
                else:
                    self.on_output(message)
            except Exception:
                pass

    async def run(self) -> None:
        self._log(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.cfg.max_cycles
        while True:
            self.cycles += 1
            self._log(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._log("[bold green]Goal complete. Halting.[/bold green]")
                break
            if max_cycles > 0 and self.cycles >= max_cycles:
                self._log(f"[yellow]Max cycles ({max_cycles}) reached. Halting.[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.tasks = []
            return

        # Carry over incomplete tasks from previous cycles
        pending_tasks = [t for t in self.tasks if t.status != "completed"]
        new_tasks = [t for t in plan.tasks if t.id not in {pt.id for pt in pending_tasks}]
        self.tasks = pending_tasks + new_tasks

        # Execute tasks respecting dependencies (parallel where possible)
        await self._execute_tasks(self.tasks)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[Any]) -> None:
        """Execute tasks, respecting dependency ordering where possible."""
        completed_ids: set[str] = set()

        def ready(task: Any) -> bool:
            return all(dep in completed_ids for dep in task.dependencies)

        remaining = list(tasks)
        while remaining:
            ready_tasks = [t for t in remaining if ready(t)]
            if not ready_tasks:
                # Circular dependency or unmet dep — run the rest anyway
                ready_tasks = remaining

            results = await asyncio.gather(
                *(WorkerAgent(task=t, client=self.client).run() for t in ready_tasks),
                return_exceptions=True,
            )

            next_remaining: list[Any] = []
            for task, result in zip(ready_tasks, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                elif isinstance(result, WorkerResult):
                    task.status = result.status
                    task.result = result.summary
                    if result.files_modified:
                        task.files = list(set(task.files + result.files_modified))
                    console.print(f"[green]Task {task.id} completed[/green]")
                    completed_ids.add(task.id)
                else:
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")

            next_remaining = [t for t in remaining if t not in ready_tasks]
            remaining = next_remaining

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        return completed >= len(self.tasks)

    def _get_tasks(self) -> list[Any]:
        return self.tasks