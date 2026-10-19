from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[Any] = []
        self._completed_ids: set[str] = set()
        self._failed_ids: set[str] = set()
        self.on_output = on_output

    def _emit(self, message: str) -> None:
        console.print(message)
        if self.on_output is not None:
            self.on_output(message)

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        self._emit(f"Furrow started. Goal: {self.goal}")

        max_cycles = self.settings.max_cycles
        while True:
            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            await self._cycle()

            if self._is_done():
                if max_cycles > 0 and self.cycles >= max_cycles:
                    self._emit("[bold green]Goal complete. Halting.[/bold green]")
                    break
                # Continue to next cycle if we still have cycles left
                # (e.g., to accumulate more planned tasks)
                pass  # Don't break, let loop check max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                self._emit(f"[yellow]Reached max cycles ({max_cycles}). Stopping.[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan: Plan = await self.planner.plan(self.goal)

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit(f"Plan: {plan.model_dump()}")

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge new tasks with existing ones, avoiding duplicates by ID
        existing_ids = {t.id for t in self.tasks}
        for task in plan.tasks:
            if task.id not in existing_ids:
                self.tasks.append(task)

        # Filter to pending tasks that haven't been attempted yet
        pending = [t for t in self.tasks if t.status == "pending"]
        if not pending:
            self._emit("[yellow]All known tasks are done or failed.[/yellow]")
            return

        max_parallel = self.settings.max_parallel_tasks
        batch = pending[:max_parallel]

        with Status(
            f"[bold yellow]Executing {len(batch)} task(s) in parallel...[/bold yellow]",
            console=console,
        ):
            worker_agents = [
                WorkerAgent(task=task, client=self.client, settings=self.settings) for task in batch
            ]
            results = await asyncio.gather(
                *[a.run() for a in worker_agents], return_exceptions=True
            )

        for task, result in zip(batch, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._failed_ids.add(task.id)
                self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._completed_ids.add(task.id)
                self._emit(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result: TestResult = await self.tester.run(self.goal, self.tasks)

        if test_result.passed:
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Update goal to focus on fixing failures
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        total = len(self.tasks)
        # Done if all tasks completed and no failures, or all tasks are terminal
        if failed > 0 and completed + failed < total:
            return False  # still pending tasks
        if failed > 0:
            return False  # have failures, try again
        return completed >= total
