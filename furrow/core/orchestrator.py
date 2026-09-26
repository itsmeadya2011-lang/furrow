from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from io import StringIO
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult, setup_logging
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger("furrow.orchestrator")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_handler: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._last_test_result: TestResult | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._output_handler = output_handler

    async def _print(self, *args: Any, **kwargs: Any) -> None:
        if self._output_handler is not None:
            buf = StringIO()
            temp = Console(file=buf, record=True, no_color=True, force_terminal=False, width=120)
            temp.print(*args, **kwargs)
            plain = temp.export_text(clear=False)
            await self._output_handler(plain)
        else:
            console.print(*args, **kwargs)

    async def run(self) -> None:
        setup_logging(self.client.settings.log_level)
        await self._print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        max_cycles = self.client.settings.max_cycles
        while True:
            if max_cycles > 0 and self.cycles >= max_cycles:
                await self._print("[yellow]Max cycles reached. Halting.[/yellow]")
                break
            self.cycles += 1
            await self._print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_start", cycle=self.cycles)
            await self._cycle()
            if self._is_done():
                await self._print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._tasks = plan.tasks
        await self._print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def run_task(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        tasks = [run_task(task) for task in plan.tasks]
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result if isinstance(result, str) else str(result)
                await self._print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self._last_test_result = test_result

        if test_result.passed:
            await self._print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._print(f"  • {failure}")
            await self._print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        completed = sum(1 for t in self._tasks if t.status == "completed")
        failed = sum(1 for t in self._tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._tasks):
            if self._last_test_result is not None:
                return self._last_test_result.passed
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks
