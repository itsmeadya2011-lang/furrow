from __future__ import annotations

import asyncio
from typing import Any, Protocol

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class ProgressCallback(Protocol):
    """Optional callback for streaming progress (e.g. to WebSocket clients)."""

    async def __call__(self, event: str, data: Any) -> None: ...


class Orchestrator:
    def __init__(
        self, goal: str, client: LLMClient | None = None, on_progress: ProgressCallback | None = None
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._current_plan: Plan | None = None
        self._test_result: TestResult | None = None
        self._on_progress = on_progress

    async def _emit(self, event: str, data: Any = None) -> None:
        if self._on_progress is not None:
            await self._on_progress(event, data)

    async def run(self) -> None:
        await self._emit("start", self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            max_cycles = self.client.settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print(
                    f"[yellow]Reached max_cycles ({max_cycles}). Stopping.[/yellow]"
                )
                await self._emit("max_cycles_reached", self.cycles)
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._emit("done", {"cycles": self.cycles})
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._current_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit("plan", plan.model_dump())

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def _bounded_run(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *(_bounded_run(task) for task in plan.tasks),
                return_exceptions=True,
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
        await self._emit("tasks_completed", [t.model_dump() for t in plan.tasks])

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self._test_result = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
        await self._emit("test_result", test_result.model_dump())

    def _is_done(self) -> bool:
        """Done when tests pass, or when the planner returns no tasks."""
        if self._test_result is not None and self._test_result.passed:
            return True
        # If no tasks were planned, there's nothing to do
        if self._current_plan is not None and not self._current_plan.tasks:
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        if self._current_plan is not None:
            return self._current_plan.tasks
        return []
