from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
logger = get_logger("furrow.orchestrator")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings if settings is not None else default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self.last_test_passed: bool = False

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        max_cycles = self.settings.max_cycles
        while True:
            self.cycles += 1
            logger.info("cycle_start", cycle=self.cycles, goal=self.goal)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                logger.info("goal_complete", cycle=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if max_cycles and self.cycles >= max_cycles:
                logger.warning(
                    "max_cycles_reached", cycles=self.cycles, max=max_cycles
                )
                console.print("[yellow]Max cycles reached. Stopping.[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self.tasks = plan.tasks
        console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )
        logger.debug("plan_created", tasks=len(plan.tasks))

        if not plan.tasks:
            logger.warning("no_tasks_planned")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        sem = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def _run(task: TaskModel) -> str:
            async with sem:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *[_run(task) for task in plan.tasks], return_exceptions=True
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                logger.error("task_failed", task=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                logger.info("task_completed", task=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self.last_test_result = test_result
        self.last_test_passed = test_result.passed
        if test_result.passed:
            logger.info("tests_passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning(
                "tests_failed",
                summary=test_result.summary,
                failures=test_result.failures,
            )
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        if any(task.status == "failed" for task in tasks):
            return False
        if not all(task.status == "completed" for task in tasks):
            return False
        return self.last_test_passed

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
