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

console = Console()
logger = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        console: Console | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.console = console if console is not None else Console()
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        max_parallel = (
            self.client.settings.max_parallel_tasks
            if self.client is not None
            else 5
        )
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(max_parallel)

    async def run(self) -> None:
        logger.info(
            "orchestrator_start",
            goal=self.original_goal,
            max_cycles=self.client.settings.max_cycles,
            max_parallel_tasks=self.client.settings.max_parallel_tasks,
        )
        self.console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}",
                title="Furrow",
            )
        )
        while True:
            self.cycles += 1
            if self._max_cycles_reached():
                logger.info(
                    "max_cycles_reached",
                    cycles=self.cycles,
                    max_cycles=self.client.settings.max_cycles,
                )
                self.console.print(
                    f"[yellow]Reached max_cycles ({self.client.settings.max_cycles}). Halting.[/yellow]"
                )
                break
            self.console.print(
                f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]"
            )
            await self._cycle()
            if self._is_done():
                self.console.print(
                    "[bold green]Goal complete. Halting.[/bold green]"
                )
                logger.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console):
            plan = await self.planner.plan(self.goal)
        self.console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )
        self.tasks = plan.tasks

        if not plan.tasks:
            self.console.print(
                "[yellow]No tasks planned. Goal may be complete.[/yellow]"
            )
            return

        with Status(
            "[bold yellow]Executing tasks in parallel...",
            console=self.console,
        ):
            results = await asyncio.gather(
                *(self._execute_task(task) for task in plan.tasks),
                return_exceptions=True,
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                logger.error("task_failed", task_id=task.id, error=str(result))
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                logger.info("task_completed", task_id=task.id)
                self.console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=self.console):
            self.last_test_result = await self.tester.run(self.goal, plan.tasks)

        test_result = self.last_test_result
        if test_result is not None and test_result.passed:
            self.console.print(
                f"[green]Tests passed: {test_result.summary}[/green]"
            )
            logger.info("tests_passed", summary=test_result.summary)
        elif test_result is not None:
            self.console.print(
                f"[red]Tests failed: {test_result.summary}[/red]"
            )
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            logger.warning(
                "tests_failed",
                summary=test_result.summary,
                failures=test_result.failures,
            )
            self.console.print(
                "[yellow]Will attempt fix in next cycle.[/yellow]"
            )
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_task(self, task: TaskModel) -> str:
        sem = self._semaphore
        assert sem is not None
        async with sem:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        if any(t.status == "failed" for t in tasks):
            return False
        if not all(t.status == "completed" for t in tasks):
            return False
        return (
            self.last_test_result is not None and self.last_test_result.passed
        )

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks

    def _max_cycles_reached(self) -> bool:
        max_cycles = self.client.settings.max_cycles
        return max_cycles > 0 and self.cycles >= max_cycles
