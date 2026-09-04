from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
log = get_logger(__name__)

ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = max_cycles if max_cycles is not None else self.client.settings.max_cycles
        self._last_tasks: list[TaskModel] = []
        self._empty_plan_count = 0
        self._last_test_passed: bool | None = None
        self._callbacks: list[ProgressCallback] = []
        self._semaphore: asyncio.Semaphore | None = None

    def add_progress_callback(self, callback: ProgressCallback) -> None:
        """Register a coroutine callback that receives (event, data)."""
        self._callbacks.append(callback)

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)
        return self._semaphore

    async def _emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        payload = data or {}
        log.info(event, **payload)
        for cb in self._callbacks:
            try:
                await cb(event, payload)
            except Exception as e:
                log.warning("progress callback failed", error=str(e))

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        await self._emit("start", {"goal": self.goal, "max_cycles": self.max_cycles})

        while True:
            if self.max_cycles and self.cycles >= self.max_cycles:
                console.print("[yellow]Max cycles reached. Halting.[/yellow]")
                await self._emit("max_cycles_reached", {"cycles": self.cycles})
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._emit("cycle_start", {"cycle": self.cycles})

            try:
                await self._cycle()
            except Exception as e:
                console.print(f"[red]Cycle {self.cycles} encountered an error: {e}[/red]")
                log.error("cycle_error", cycle=self.cycles, error=str(e))
                await self._emit("cycle_error", {"cycle": self.cycles, "error": str(e)})

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._emit("complete", {"cycles": self.cycles})
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit("plan", {"plan": plan.model_dump()})

        self._last_tasks = plan.tasks

        if not plan.tasks:
            self._empty_plan_count += 1
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._emit("no_tasks", {"consecutive_empty": self._empty_plan_count})
            return

        self._empty_plan_count = 0

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                self._run_task_with_limit(task)
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._emit("task_failed", {"task_id": task.id, "error": str(result)})
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._emit("task_completed", {"task_id": task.id, "result": result})

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            await self._emit("tests_passed", {"summary": test_result.summary})
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            await self._emit("tests_failed", {"summary": test_result.summary, "failures": test_result.failures})
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        self._last_test_passed = test_result.passed

    async def _run_task_with_limit(self, task: TaskModel) -> str:
        async with self.semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        """Return True when the goal is complete, max cycles exceeded, or the planner
        has returned empty plans repeatedly (giving up)."""
        if self._empty_plan_count >= 3:
            return True

        tasks = self._get_tasks()
        if not tasks:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            # All tasks completed — goal is only done if tests also passed.
            return self._last_test_passed is True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self._last_tasks
