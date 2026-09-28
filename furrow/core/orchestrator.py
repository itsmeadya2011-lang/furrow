from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import structlog
from rich.console import Console

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger("furrow")

console = Console()

OutputCallback = Callable[[str], Awaitable[None]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output: OutputCallback | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient()
        self.settings: Settings = self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.completed_tasks: list[Any] = []
        self.failed_tasks: list[Any] = []
        self.last_plan: Any = None
        self.last_test_passed: bool | None = None
        self.output = output
        self._semaphore: asyncio.Semaphore | None = None

    async def _emit(self, msg: str) -> None:
        if self.output is not None:
            await self.output(msg)
        else:
            console.print(msg)

    async def run(self) -> None:
        await self._emit(f"[bold green]Furrow[/bold green] | Goal: {self.goal}")
        max_cycles = self.settings.max_cycles or 0  # 0 == unlimited
        try:
            while True:
                self.cycles += 1
                if max_cycles and self.cycles > max_cycles:
                    await self._emit("[yellow]Max cycles reached. Stopping.[/yellow]")
                    break
                await self._emit(f"[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                await self._cycle()
                if self._is_done():
                    await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                    break
                if self.last_plan is not None and not self.last_plan.tasks:
                    await self._emit("[yellow]No further tasks to plan. Stopping.[/yellow]")
                    break
        except asyncio.CancelledError:
            logger.info("orchestrator_cancelled")
            raise

    async def _cycle(self) -> None:
        plan = await self.planner.plan(self.goal)
        self.last_plan = plan
        plan_dump = json.dumps(plan.model_dump(), indent=2, default=str)
        await self._emit(f"[bold]Plan:[/bold]\n{plan_dump}")

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.last_test_passed = True
            return

        logger.info("orchestrator_cycle_start", cycle=self.cycles, tasks=len(plan.tasks))
        self._semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        worker_tasks = [self._run_worker(t) for t in plan.tasks]
        results: list[Any] = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.failed_tasks.append(task)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.completed_tasks.append(task)
                await self._emit(f"[green]Task {task.id} completed[/green]")

        test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self.last_test_passed = test_result.passed
        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _run_worker(self, task: Any) -> str:
        assert self._semaphore is not None
        async with self._semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        if self.last_test_passed is True:
            return True
        return bool(self.last_plan is not None and not self.last_plan.tasks)
