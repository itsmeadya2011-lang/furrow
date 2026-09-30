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
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.settings: Settings = settings
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_plan: Plan | None = None
        self.last_test_result: TestResult | None = None
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def _sem(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        return self._semaphore

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        logger.info(
            "furrow.start",
            goal=self.goal,
            max_cycles=self.settings.max_cycles,
            max_parallel=self.settings.max_parallel_tasks,
        )
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done() or self._max_cycles_reached():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("furrow.complete", cycles=self.cycles, tasks=len(self.tasks))
                break

    def _max_cycles_reached(self) -> bool:
        return bool(self.settings.max_cycles and self.cycles >= self.settings.max_cycles)

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self.last_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("furrow.plan", tasks=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.tasks = []
            return

        # Store tasks on self so completion logic and tests can introspect them.
        self.tasks = plan.tasks

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [self._run_task(task) for task in self.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(self.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("furrow.task_failed", task=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("furrow.task_completed", task=task.id)

        with Status("[bold yellow]Testing...", console=console):
            self.last_test_result = await TesterAgent(client=self.client).run(
                self.goal, self.tasks
            )

        if self.last_test_result.passed:
            console.print(f"[green]Tests passed: {self.last_test_result.summary}[/green]")
            logger.info("furrow.tests_passed")
        else:
            console.print(f"[red]Tests failed: {self.last_test_result.summary}[/red]")
            for failure in self.last_test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(self.last_test_result.failures)
            logger.warning(
                "furrow.tests_failed", failures=len(self.last_test_result.failures)
            )

    async def _run_task(self, task: TaskModel) -> str:
        async with self._sem:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        # No plan yet, or an empty plan, means there is nothing left to do.
        if self.last_plan is None or not self.last_plan.tasks:
            return True
        completed = [t for t in self.tasks if t.status == "completed"]
        failed = [t for t in self.tasks if t.status == "failed"]
        if failed:
            # A task failed; the next cycle will re-plan a fix (bounded by max_cycles).
            return False
        return bool(
            completed
            and len(completed) >= len(self.tasks)
            and self.last_test_result
            and self.last_test_result.passed
        )
