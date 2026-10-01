from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TaskStatus, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

logger = structlog.get_logger(__name__)

OutputHandler = Callable[[str], Awaitable[None]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        console: Console | None = None,
        output_handler: OutputHandler | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or (client.settings if client else default_settings)
        self.client = client or LLMClient(settings=self.settings)
        self.console = console or Console()
        self.output_handler: OutputHandler | None = output_handler
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self._all_tasks: list[TaskModel] = []
        self._last_test_result: TestResult | None = None
        self._cycle_had_tasks: bool = False

    async def _emit(self, message: str) -> None:
        """Send a message to the console (if present) and/or the output handler."""
        self.console.print(message)
        if self.output_handler is not None:
            await self.output_handler(message)

    async def run(self) -> None:
        await self._emit(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                await self._emit("[yellow]Max cycles reached. Stopping.[/yellow]")
                logger.info("max_cycles_reached after %s cycles", self.cycles)
                break

            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            try:
                await self._cycle()
            except Exception as e:
                logger.exception("cycle_error: %s", e)
                await self._emit(f"[red]Cycle {self.cycles} failed: {e}[/red]")

            if self._is_done():
                if self._last_test_result and self._last_test_result.passed:
                    await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                else:
                    await self._emit("[yellow]Stopping (max cycles or no progress).[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console):
            plan = await self.planner.plan(self.goal)

        await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        # Track whether this cycle produced any actionable tasks
        self._cycle_had_tasks = bool(plan.tasks)

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Track tasks for status visibility
        for task in plan.tasks:
            if not any(t.id == task.id for t in self._all_tasks):
                self._all_tasks.append(task)

        # Execute tasks with parallelism limit
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def _run_task(task: TaskModel) -> str:
            task.status = TaskStatus.IN_PROGRESS
            async with semaphore:
                result = await WorkerAgent(task=task, client=self.client).run()
            task.status = TaskStatus.COMPLETED
            task.result = result
            return result

        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            raw_results = await asyncio.gather(
                *[_run_task(t) for t in plan.tasks], return_exceptions=True
            )

        for task, result in zip(plan.tasks, raw_results):
            if isinstance(result, Exception):
                task.status = TaskStatus.FAILED
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = TaskStatus.COMPLETED
                task.result = result
                await self._emit(f"[green]Task {task.id} completed[/green]")

        # Test
        with Status("[bold yellow]Testing...", console=self.console):
            test_result = await self.tester.run(self.goal, plan.tasks)

        self._last_test_result = test_result

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = (
                f"Fix failing tests. Original goal: {self.goal}\n"
                + "\n".join(test_result.failures)
            )

    def _is_done(self) -> bool:
        """Check if the orchestration loop should terminate."""
        tests_passed = self._last_test_result is not None and self._last_test_result.passed
        max_reached = self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles
        stuck = not self._cycle_had_tasks and self._last_test_result is None
        return tests_passed or max_reached or stuck
