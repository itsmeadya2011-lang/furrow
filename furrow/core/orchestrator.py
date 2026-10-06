from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    """Runs the Furrow infinite development loop until the goal is complete."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings = settings,
    ) -> None:
        self.goal = goal
        self.settings = settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._last_test: TestResult | None = None
        self.on_event: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None

    async def _emit(self, event: str, **data: Any) -> None:
        if self.on_event is not None:
            await self.on_event(event, data)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        await self._emit("start", goal=self.goal)
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._emit("complete")
                break
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print("[yellow]Max cycles reached. Halting.[/yellow]")
                await self._emit("max_cycles_reached", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        await self._emit("cycle_started", cycle=self.cycles, goal=self.goal)

        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit("plan_ready", plan=plan.model_dump())

        # Persist task state so it survives across cycles for the done-check.
        self._tasks = plan.tasks

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._emit("no_tasks")
            return

        sem = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def _bounded(task: TaskModel) -> str:
            async with sem:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *[_bounded(task) for task in plan.tasks],
                return_exceptions=True,
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._emit("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._emit("task_completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=console):
            self._last_test = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if self._last_test.passed:
            console.print(f"[green]Tests passed: {self._last_test.summary}[/green]")
            await self._emit("tests_passed", summary=self._last_test.summary)
        else:
            console.print(f"[red]Tests failed: {self._last_test.summary}[/red]")
            for failure in self._last_test.failures:
                console.print(f"  • {failure}")
            await self._emit("tests_failed", failures=self._last_test.failures)
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(self._last_test.failures)

        await self._emit("cycle_finished", cycle=self.cycles)

    def _is_done(self) -> bool:
        # No tasks at all means there is nothing left to do.
        if not self._tasks:
            return True
        # Any failed task means we should retry (the loop continues).
        if any(t.status == "failed" for t in self._tasks):
            return False
        # Not all tasks completed yet.
        if not all(t.status == "completed" for t in self._tasks):
            return False
        # Tests must have validated success before we consider the goal done.
        if self._last_test is None:
            return False
        return self._last_test.passed

    def _get_tasks(self) -> list[Any]:
        return self._tasks