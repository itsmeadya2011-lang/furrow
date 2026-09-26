from __future__ import annotations

import asyncio
from collections.abc import Callable

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tasks: list[TaskModel] = []
        self.cycles = 0
        self.on_progress = on_progress

    async def run(self, on_progress: Callable[[str], None] | None = None) -> None:
        if on_progress is not None:
            self.on_progress = on_progress

        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        self._log("Furrow started")

        while True:
            max_cycles = self.client.settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                self._log(f"Reached max_cycles ({max_cycles}); halting.")
                logger.info("max_cycles_reached", max_cycles=max_cycles)
                break

            self.cycles += 1
            self._log(f"Starting cycle {self.cycles}")
            logger.info("cycle_start", cycle=self.cycles, goal=self.goal)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            await self._cycle()

            if self._is_done():
                self._log("Goal complete. Halting.")
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycle=self.cycles)
                break

            logger.info("cycle_end", cycle=self.cycles, done=False)

    async def _cycle(self) -> None:
        self._log("Planning...")
        logger.info("planning", goal=self.goal)
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.tasks = list(plan.tasks)

        self._log(f"Plan created with {len(plan.tasks)} task(s)")
        logger.info("plan_created", num_tasks=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._log("No tasks planned; goal may be complete.")
            return

        semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)
        self._log(
            f"Executing {len(plan.tasks)} task(s) "
            f"(max parallel: {self.client.settings.max_parallel_tasks})"
        )
        logger.info(
            "execution_start",
            num_tasks=len(plan.tasks),
            max_parallel=self.client.settings.max_parallel_tasks,
        )

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [self._run_task_bounded(task, semaphore) for task in plan.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                self._log(f"Task {task.id} failed: {result}")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                self._log(f"Task {task.id} completed")
                logger.info("task_completed", task_id=task.id)

        self._log("Testing...")
        logger.info("testing_start", goal=self.goal)
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._log(f"Tests passed: {test_result.summary}")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            self._log(f"Tests failed: {test_result.summary}")
            logger.error(
                "tests_failed",
                summary=test_result.summary,
                failures=test_result.failures,
            )
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self._log("Will attempt fix in next cycle.")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        logger.info("cycle_end", cycle=self.cycles)

    async def _run_task_bounded(self, task: TaskModel, semaphore: asyncio.Semaphore) -> str:
        async with semaphore:
            logger.info("task_start", task_id=task.id)
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks

    def _log(self, msg: str) -> None:
        console.print(msg)
        if self.on_progress is not None:
            self.on_progress(msg)
        logger.info(msg)
