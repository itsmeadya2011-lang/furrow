from __future__ import annotations

import asyncio
from typing import Callable

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, TaskModel
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.output_callback = output_callback

    def _output(self, text: str) -> None:
        """Print plain text to the Rich console and forward to callback if set."""
        console.print(text)
        if self.output_callback is not None:
            self.output_callback(text)

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        logger.info("orchestrator.start", goal=self.goal)
        while True:
            self.cycles += 1
            logger.info("orchestrator.cycle", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._output(f"Cycle {self.cycles}")
            await self._cycle()
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                self._output("[bold yellow]Max cycles reached[/bold yellow]")
                logger.info("orchestrator.max_cycles_reached", cycles=self.cycles)
                break
            if self._is_done():
                self._output("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("orchestrator.done")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        logger.info("orchestrator.plan", tasks=len(plan.tasks))
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.tasks = plan.tasks

        if not self.tasks:
            self._output("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def run_task(task: TaskModel) -> TaskModel:
            async with semaphore:
                try:
                    result = await WorkerAgent(task=task, client=self.client).run()
                    task.status = "completed"
                    task.result = result
                    self._output(f"[green]Task {task.id} completed[/green]")
                    logger.info("orchestrator.task_completed", task_id=task.id)
                except Exception as exc:
                    task.status = "failed"
                    task.result = str(exc)
                    self._output(f"[red]Task {task.id} failed: {exc}[/red]")
                    logger.error("orchestrator.task_failed", task_id=task.id, error=str(exc))
                return task

        worker_coroutines = [run_task(task) for task in self.tasks]
        await asyncio.gather(*worker_coroutines)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        logger.info("orchestrator.test", passed=test_result.passed)
        if test_result.passed:
            self._output(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._output(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._output(f"  • {failure}")
            self._output("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
