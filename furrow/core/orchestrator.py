from __future__ import annotations

import asyncio
from collections.abc import Callable

import structlog
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        reporter: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.reporter = reporter
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self.cycles = 0

    async def _report(self, msg: str | RenderableType) -> None:
        console.print(msg)
        if self.reporter is not None:
            if isinstance(msg, str):
                self.reporter(msg)
            else:
                with console.capture() as cap:
                    console.print(msg)
                self.reporter(cap.get())

    async def run(self) -> None:
        await self._report(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"
            )
        )
        while True:
            self.cycles += 1
            await self._report(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                await self._report("[bold green]Goal complete. Halting.[/bold green]")
                break
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                await self._report(
                    "[bold red]Max cycles reached. Halting.[/bold red]"
                )
                break

    async def _cycle(self) -> None:
        logger.info("planning", goal=self.goal, cycle=self.cycles)
        with Status("[bold yellow]Planning...", console=console):
            plan: Plan = await self.planner.plan(self.goal)
        await self._report(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        self.tasks = plan.tasks

        if not plan.tasks:
            await self._report(
                "[yellow]No tasks planned. Goal may be complete.[/yellow]"
            )
            return

        logger.info("executing_tasks", count=len(plan.tasks))
        semaphore = asyncio.Semaphore(settings.max_parallel_tasks)

        async def _run_with_semaphore(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        task_coros = [_run_with_semaphore(task) for task in plan.tasks]
        results = await asyncio.gather(*task_coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, str):
                task.status = "completed"
                task.result = result
                logger.debug("task_completed", task_id=task.id)
                await self._report(f"[green]Task {task.id} completed[/green]")
            else:
                task.status = "failed"
                task.result = str(result)
                logger.error("task_failed", task_id=task.id, error=str(result))
                await self._report(f"[red]Task {task.id} failed: {result}[/red]")

        logger.info("testing", goal=self.goal)
        with Status("[bold yellow]Testing...", console=console):
            self.last_test_result = await TesterAgent(
                client=self.client
            ).run(self.goal, plan.tasks)

        if self.last_test_result.passed:
            await self._report(
                f"[green]Tests passed: {self.last_test_result.summary}[/green]"
            )
        else:
            await self._report(
                f"[red]Tests failed: {self.last_test_result.summary}[/red]"
            )
            for failure in self.last_test_result.failures:
                await self._report(f"  • {failure}")
            await self._report(
                "[yellow]Will attempt fix in next cycle.[/yellow]"
            )
            self.goal = (
                "Fix failing tests:\n" + "\n".join(self.last_test_result.failures)
            )

    def _is_done(self) -> bool:
        if not self.tasks:
            return True
        if any(t.status != "completed" for t in self.tasks):
            return False
        return self.last_test_result is not None and self.last_test_result.passed

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
