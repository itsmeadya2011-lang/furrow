from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger(__name__)

Reporter = Callable[[str], Awaitable[None]]

__all__ = ["Orchestrator", "Reporter"]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        reporter: Reporter | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.reporter: Reporter | None = reporter
        self.cycles = 0
        self._tasks: list[Any] = []
        self._done = False

    async def run(self) -> None:
        from furrow.config import settings

        logger.info("orchestrator_start", goal=self.goal, max_cycles=settings.max_cycles)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))

        try:
            while not self._done:
                self.cycles += 1

                if settings.max_cycles > 0 and self.cycles > settings.max_cycles:
                    logger.info("max_cycles_reached", cycles=self.cycles)
                    await self._report(
                        f"[yellow]Reached max_cycles ({settings.max_cycles}). Stopping.[/yellow]"
                    )
                    break

                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                await self._report(f"Cycle {self.cycles}: Planning...")
                await self._cycle()

                if self._is_done():
                    logger.info("goal_complete", cycles=self.cycles)
                    await self._report("[bold green]Goal complete. Halting.[/bold green]")
                    break
        finally:
            await self.client.close()

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._report(f"Plan created: {plan.rationale}")

        if not plan.tasks:
            logger.info("no_tasks_planned")
            await self._report("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._done = True
            return

        # Track tasks for this cycle
        self._tasks = list(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            workers = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*workers, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.mark_failed(str(result))
                logger.error("task_failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._report(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.mark_completed(result)
                logger.info("task_completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._report(f"[green]Task {task.id} completed[/green]")

        # Update tracked tasks with final status
        self._tasks = plan.tasks

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            logger.info("tests_passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            await self._report(f"[green]Tests passed: {test_result.summary}[/green]")
            self._done = True
        else:
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
                await self._report(f"[red]  • {failure}[/red]")
            await self._report("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _report(self, message: str) -> None:
        """Send a message to the reporter if available."""
        if self.reporter is not None:
            await self.reporter(message)

    def _is_done(self) -> bool:
        if self._done:
            return True
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self._tasks
