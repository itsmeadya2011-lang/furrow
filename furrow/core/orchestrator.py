from __future__ import annotations

import asyncio
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []

    async def run(self) -> None:
        logger.info("orchestrator.start", goal=self.goal)
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            self.cycles += 1
            logger.info("orchestrator.cycle", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

            if self._is_done():
                logger.info("orchestrator.complete")
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

            max_cycles = getattr(self.client.settings, "max_cycles", 0)
            if max_cycles > 0 and self.cycles >= max_cycles:
                logger.info("orchestrator.max_cycles_reached", max_cycles=max_cycles)
                console.print(
                    f"[bold yellow]Reached max_cycles ({max_cycles}). Halting.[/bold yellow]"
                )
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        self.tasks = list(plan.tasks) if plan.tasks else []

        if not self.tasks:
            console.print(
                "[yellow]No tasks planned. Goal may be complete.[/yellow]"
            )
            return

        max_parallel = getattr(self.client.settings, "max_parallel_tasks", 0)
        if max_parallel <= 0:
            max_parallel = 1
        sem = asyncio.Semaphore(max_parallel)

        async def _run_worker(task: Any) -> Any:
            async with sem:
                agent = WorkerAgent(task=task, client=self.client)
                return await agent.run()

        with Status(
            "[bold yellow]Executing tasks in parallel...", console=console
        ):
            coros = [_run_worker(task) for task in self.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(self.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                logger.error("task.failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                logger.info("task.completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")

        self._display_cycle_summary()

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(
                self.goal, self.tasks
            )

        if test_result.passed:
            logger.info("tests.passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("tests.failed", summary=test_result.summary)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print(
                "[yellow]Will attempt fix in next cycle.[/yellow]"
            )
            self.goal = (
                "Fix failing tests:\n"
                + "\n".join(test_result.failures)
            )

    def _display_cycle_summary(self) -> None:
        tasks = self._get_tasks()
        completed = sum(
            1 for t in tasks if getattr(t, "status", "pending") == "completed"
        )
        failed = sum(
            1 for t in tasks if getattr(t, "status", "pending") == "failed"
        )
        pending = sum(
            1
            for t in tasks
            if getattr(t, "status", "pending") not in ("completed", "failed")
        )
        console.print(
            f"[bold cyan]Cycle {self.cycles} summary:[/bold cyan] "
            f"completed={completed} failed={failed} pending={pending}"
        )

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        completed = sum(
            1 for t in tasks if getattr(t, "status", "pending") == "completed"
        )
        failed = sum(
            1 for t in tasks if getattr(t, "status", "pending") == "failed"
        )
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        return self.tasks
