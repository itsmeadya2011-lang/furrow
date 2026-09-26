from __future__ import annotations

import asyncio

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

logger = structlog.get_logger()

_default_console = Console()


class Orchestrator:
    """Runs the infinite Furrow development loop: plan, execute, test, repeat."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.console = console or _default_console
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.test_result: TestResult | None = None

    async def run(self) -> None:
        self.console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        max_cycles = self.client.settings.max_cycles
        while True:
            self.cycles += 1
            self.console.print(
                f"\n[bold cyan]Cycle {self.cycles} / {max_cycles or 'unlimited'}[/bold cyan]"
            )
            await self._cycle()
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                break
            if max_cycles > 0 and self.cycles >= max_cycles:
                self.console.print(f"[yellow]Reached max_cycles ({max_cycles}). Stopping.[/yellow]")
                logger.info("max_cycles_reached", cycles=self.cycles, max_cycles=max_cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console):
            plan = await self.planner.plan(self.goal)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.tasks = []
            return

        self.tasks = plan.tasks

        # Limit parallelism using the configured max_parallel_tasks
        max_parallel = self.client.settings.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_with_semaphore(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            coros = [run_with_semaphore(task) for task in plan.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=self.console):
            self.test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if self.test_result.passed:
            self.console.print(f"[green]Tests passed: {self.test_result.summary}[/green]")
        else:
            self.console.print(f"[red]Tests failed: {self.test_result.summary}[/red]")
            for failure in self.test_result.failures:
                self.console.print(f"  - {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(self.test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return True
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed < len(self.tasks):
            return False
        # All tasks completed — verify test results
        return not (self.test_result and not self.test_result.passed)
