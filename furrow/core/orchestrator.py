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
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()

# Type alias for an async event callback
EventCallback = Callable[[dict[str, Any]], Awaitable[None]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int | None = None,
        max_parallel: int | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = max_cycles if max_cycles is not None else self.client.settings.max_cycles
        self.max_parallel = max_parallel if max_parallel is not None else self.client.settings.max_parallel_tasks
        self._current_plan: Plan | None = None
        self._all_tasks: list[TaskModel] = []
        self._on_event: EventCallback | None = on_event

    async def _emit(self, event: dict[str, Any]) -> None:
        if self._on_event is not None:
            await self._on_event(event)

    async def run(self) -> list[TaskModel]:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        await self._emit({"type": "start", "goal": self.goal})
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._emit({"type": "cycle_start", "cycle": self.cycles})
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._emit({"type": "complete", "task_count": len(self._all_tasks)})
                break
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                console.print(f"[yellow]Reached max_cycles ({self.max_cycles}). Halting.[/yellow]")
                await self._emit({"type": "max_cycles_reached", "cycles": self.cycles})
                break
        return self._all_tasks

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit({"type": "plan", "plan": plan.model_dump()})

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._current_plan = plan
            return

        self._current_plan = plan
        self._all_tasks.extend(plan.tasks)
        await self._emit({"type": "tasks_scheduled", "count": len(plan.tasks)})

        semaphore = asyncio.Semaphore(self.max_parallel)

        async def _run_task(task: TaskModel) -> Any:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [_run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._emit({"type": "task_failed", "task_id": task.id, "error": str(result)})
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._emit({"type": "task_completed", "task_id": task.id})

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )
        await self._emit({"type": "test_result", "test_result": test_result.model_dump()})

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                console.print("[yellow]Cannot attempt fix: max_cycles reached.[/yellow]")
                return
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if self.max_cycles > 0 and self.cycles >= self.max_cycles:
            return False  # let the run loop handle the final halt message
        tasks = self._get_tasks()
        if not tasks:
            return True
        failed = sum(1 for t in tasks if t.status == "failed")
        completed = sum(1 for t in tasks if t.status == "completed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        if self._current_plan is not None and self._current_plan.tasks:
            return self._current_plan.tasks
        return self._all_tasks
