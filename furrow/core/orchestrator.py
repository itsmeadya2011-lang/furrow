from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

logger = structlog.get_logger(__name__)

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        log_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._tests_passed: bool = False
        self.max_cycles = self.client.settings.max_cycles
        self._log_callback = log_callback

    def _strip_ansi(self, text: str) -> str:
        return re.sub(r"\x1b\[[0-9;]*m", "", text)

    def _emit(self, text: str) -> None:
        if self._log_callback is not None:
            clean = self._strip_ansi(text)
            asyncio.create_task(self._log_callback(clean))
        else:
            console.print(text)

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                self._emit("[yellow]Max cycles reached. Stopping.[/yellow]")
                break
            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                break
            self._emit("[yellow]Cycle complete. Continuing to next cycle...[/yellow]")

    async def _cycle(self) -> None:
        logger.info("planning", goal=self.goal)
        with Status("[bold yellow]Planning...", console=console):
            plan: Plan = await self.planner.plan(self.goal)
        self.tasks = plan.tasks

        with console.capture() as cap:
            console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit(cap.get())

        if not self.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        logger.info("executing_tasks", count=len(self.tasks))
        semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def _run_task(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        tasks_to_run = [_run_task(t) for t in self.tasks]
        results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

        for task, result in zip(self.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit(f"[red]Task {task.id} failed: {result}[/red]")
                logger.warning("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                self._emit(f"[green]Task {task.id} completed[/green]")

        logger.info("testing", goal=self.goal)
        with Status("[bold yellow]Testing...", console=console):
            tester_agent = TesterAgent(client=self.client)
            test_result: TestResult = await tester_agent.run(self.goal, self.tasks)

        self._tests_passed = test_result.passed

        if test_result.passed:
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        logger.info("cycle_finished", passed=test_result.passed)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        if any(t.status == "failed" for t in tasks):
            return False
        if all(t.status == "completed" for t in tasks):
            return self._tests_passed
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
