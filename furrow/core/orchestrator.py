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
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger("orchestrator")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        verbose: bool = False,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self.last_test_result: TestResult | None = None
        self.verbose = verbose
        self._semaphore = asyncio.Semaphore(settings.max_parallel_tasks)

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("starting_cycle", cycle=self.cycles, goal=self.goal)

            if settings.max_cycles > 0 and self.cycles > settings.max_cycles:
                console.print(
                    f"[yellow]Reached max_cycles ({settings.max_cycles}). Stopping.[/yellow]"
                )
                logger.warning(
                    "max_cycles_reached",
                    cycle=self.cycles,
                    max_cycles=settings.max_cycles,
                )
                break

            await self._cycle()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycle=self.cycles)
                break

    async def _cycle(self) -> None:
        plan_context = self.goal
        if self.last_test_result and not self.last_test_result.passed:
            failures = self.last_test_result.failures
            if failures:
                failure_context = "\n".join(failures)
                plan_context = f"{self.goal}\n\nPrevious failures to address:\n{failure_context}"

        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(plan_context)
        self.plan = plan

        if self.verbose:
            console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        else:
            logger.debug("plan_created", tasks=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [self._run_worker(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                if self.verbose:
                    console.print(f"  [dim]{result}[/dim]")
                logger.info("task_completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self.last_test_result = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning(
                "tests_failed",
                summary=test_result.summary,
                failures=test_result.failures,
            )

    async def _run_worker(self, task: TaskModel) -> str:
        async with self._semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        if self.plan is None:
            return []
        return self.plan.tasks
