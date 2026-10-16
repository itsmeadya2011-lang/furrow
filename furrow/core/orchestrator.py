from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    """Runs the infinite Furrow development loop.

    Each cycle: plan -> execute (respecting dependencies and parallelism) ->
    test -> report -> repeat.  Task state persists across cycles so the
    planner can see what was already completed or failed.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        emit: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.settings = settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.emit = emit

    async def _emit(self, message: str, style: str = "") -> None:
        """Print to console and optionally forward to an emit callback (Web UI)."""
        if style:
            console.print(f"[{style}]{message}[/{style}]")
        else:
            console.print(message)
        if self.emit:
            if asyncio.iscoroutinefunction(self.emit):
                await self.emit(message)
            else:
                self.emit(message)

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        if self.emit:
            await self.emit(f"Furrow started. Goal: {self.goal}")

        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                await self._emit(
                    f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]",
                    "yellow",
                )
                break

            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]", "cyan")
            await self._cycle()
            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]", "green")
                break

    async def _cycle(self) -> None:
        # --- Plan ---
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(
                self.goal, previous_tasks=self.all_tasks
            )

        # Track new tasks across cycles
        for task in plan.tasks:
            self.all_tasks.append(task)

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        if self.emit:
            await self.emit(f"Plan: {plan.rationale}")

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]", "yellow")
            return

        # --- Execute (respecting dependencies + max_parallel_tasks) ---
        await self._execute_tasks(plan.tasks)

        # --- Test ---
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]", "green")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]", "red")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit(
                "[yellow]Will attempt fix in next cycle.[/yellow]", "yellow"
            )
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[TaskModel]) -> None:
        """Execute tasks in dependency-aware waves, limited by max_parallel_tasks."""
        pending: dict[str, TaskModel] = {t.id: t for t in tasks}
        completed_ids: set[str] = set()
        failed_ids: set[str] = set()

        while pending:
            ready = [
                t
                for t in pending.values()
                if all(dep in completed_ids for dep in t.dependencies)
                and not any(dep in failed_ids for dep in t.dependencies)
            ]

            if not ready:
                # Remaining tasks are blocked by failed or circular dependencies.
                for t in pending.values():
                    t.status = "blocked"
                    t.result = "Blocked by unresolved dependencies"
                    await self._emit(f"[yellow]Task {t.id} blocked (dependency failed)[/yellow]", "yellow")
                break

            batch = ready[: self.settings.max_parallel_tasks]

            workers = [
                WorkerAgent(task=t, client=self.client).run() for t in batch
            ]
            results = await asyncio.gather(*workers, return_exceptions=True)

            for task, result in zip(batch, results):
                pending.pop(task.id)
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    failed_ids.add(task.id)
                    await self._emit(f"[red]Task {task.id} failed: {result}[/red]", "red")
                else:
                    task.status = "completed"
                    task.result = result
                    completed_ids.add(task.id)
                    await self._emit(f"[green]Task {task.id} completed[/green]", "green")

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        # Any failed or blocked task means the goal is not yet complete.
        if any(t.status in ("failed", "blocked") for t in self.all_tasks):
            return False
        return all(t.status == "completed" for t in self.all_tasks)
