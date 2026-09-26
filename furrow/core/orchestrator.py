from __future__ import annotations

import asyncio
import logging

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, default_settings.log_level.upper(), logging.INFO)
    ),
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger("furrow.orchestrator")

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.console: Console = console if console is not None else Console()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self._semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

    async def run(self) -> None:
        self.console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        logger.info("orchestrator_start", goal=self.goal, max_cycles=self.settings.max_cycles)
        while True:
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_start", cycle=self.cycles, goal=self.goal)
            await self._cycle()
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self.console.print("[yellow]Reached max_cycles. Halting.[/yellow]")
                logger.info("max_cycles_reached", cycles=self.cycles)
                break
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console):
            plan = await self.planner.plan(self.goal)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", cycle=self.cycles, num_tasks=len(plan.tasks))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned", cycle=self.cycles)
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            results = await asyncio.gather(
                *(self._run_task(task) for task in plan.tasks), return_exceptions=True
            )

        failed_tasks: list[TaskModel] = []
        for task, result in zip(plan.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                failed_tasks.append(task)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.warning("task_failed", cycle=self.cycles, task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", cycle=self.cycles, task_id=task.id)

        self.all_tasks.extend(plan.tasks)

        if failed_tasks:
            self.goal = "Fix failed tasks:\n" + "\n".join(
                f"- {t.id}: {t.description}" for t in failed_tasks
            )
            logger.warning("focusing_on_failed_tasks", failed_ids=[t.id for t in failed_tasks])

        with Status("[bold yellow]Testing...", console=self.console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", cycle=self.cycles, summary=test_result.summary)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        logger.info("cycle_complete", cycle=self.cycles, done=self._is_done())

    async def _run_task(self, task: TaskModel) -> str:
        async with self._semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        if any(t.status == "failed" for t in tasks):
            return False
        return all(t.status == "completed" for t in tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
