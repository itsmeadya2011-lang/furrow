from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from structlog import get_logger

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.llm import LLMClient

console = Console()
logger = get_logger()

ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int | None = None,
        max_parallel_tasks: int | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.settings = self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._failed_tasks: list[TaskModel] = []
        self._last_test_passed: bool | None = None
        self._max_cycles = max_cycles if max_cycles is not None else self.settings.max_cycles
        self._max_parallel_tasks = (
            max_parallel_tasks
            if max_parallel_tasks is not None
            else self.settings.max_parallel_tasks
        )
        self._semaphore: asyncio.Semaphore | None = None
        self.on_progress = on_progress

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_parallel_tasks)
        return self._semaphore

    @property
    def all_tasks(self) -> list[TaskModel]:
        return self._tasks + self._failed_tasks

    @property
    def completed_tasks(self) -> list[TaskModel]:
        return [t for t in self._tasks if t.status == "completed"]

    @property
    def failed_tasks(self) -> list[TaskModel]:
        return [t for t in self._tasks if t.status == "failed"]

    async def emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        if self.on_progress is not None:
            await self.on_progress(event, data or {})

    async def run(self) -> None:
        await self.emit("start", {"goal": self.goal})
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self._max_cycles > 0 and self.cycles >= self._max_cycles:
                logger.info("max_cycles_reached", cycles=self.cycles)
                await self.emit("max_cycles_reached", {"cycles": self.cycles})
                console.print(f"[yellow]Max cycles ({self._max_cycles}) reached. Stopping.[/yellow]")
                break

            self.cycles += 1
            logger.info("cycle_start", cycle=self.cycles, goal=self.goal)
            await self.emit("cycle_start", {"cycle": self.cycles})

            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                await self.emit("complete", {"cycles": self.cycles})
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self.emit("plan", {"cycle": self.cycles, "plan": plan.model_dump()})

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self.emit("no_tasks", {"cycle": self.cycles})
            return

        self._tasks = plan.tasks
        self._failed_tasks = []

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_coros = [
                self._run_worker_with_limit(task) for task in plan.tasks
            ]
            results = await asyncio.gather(*worker_coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._failed_tasks.append(task)
                logger.error("task_failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = str(result)
                logger.info("task_completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")

        await self.emit("execution_complete", {"cycle": self.cycles, "tasks": [t.model_dump() for t in plan.tasks]})

        with Status("[bold yellow]Testing...", console=console):
            test_result = await self.tester.run(self.goal, plan.tasks)

        await self.emit("test", {"cycle": self.cycles, "result": test_result.model_dump()})

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  - {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        self._last_test_passed = test_result.passed

    async def _run_worker_with_limit(self, task: TaskModel) -> str:
        async with self.semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        completed = sum(1 for t in self._tasks if t.status == "completed")
        failed = sum(1 for t in self._tasks if t.status == "failed")
        if failed > 0:
            return False
        if self._last_test_passed is False:
            return False
        return len(self._tasks) > 0 and completed >= len(self._tasks)
