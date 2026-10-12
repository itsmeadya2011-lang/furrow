from __future__ import annotations

import asyncio
from contextlib import suppress

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient
from furrow.logging import configure_logging, get_logger

configure_logging()
console = Console()
logger = get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []

    async def run(self) -> None:
        logger.info("Starting Furrow orchestrator", goal=self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                logger.warning("Max cycles reached", max_cycles=self.settings.max_cycles)
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                break

            self.cycles += 1
            logger.info("Starting cycle", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                logger.info("Goal complete", cycles=self.cycles, total_tasks=len(self.all_tasks))
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        logger.info("Planning cycle", cycle=self.cycles)
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        logger.info("Plan received", tasks_count=len(plan.tasks))
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.warning("No tasks planned")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Track tasks for this cycle
        cycle_tasks = plan.tasks
        self.all_tasks.extend(cycle_tasks)

        # Execute with concurrency limit
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def run_with_semaphore(task: TaskModel) -> str | Exception:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client, settings=self.settings).run()

        logger.info("Executing tasks", count=len(cycle_tasks), max_parallel=self.settings.max_parallel_tasks)
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [run_with_semaphore(task) for task in cycle_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(cycle_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                logger.error("Task failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                logger.info("Task completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")

        logger.info("Running tests")
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await self.tester.run(self.goal, cycle_tasks)

        if test_result.passed:
            logger.info("Tests passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("Tests failed", failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")

            # Generate fix and spawn a fix worker
            console.print("[yellow]Generating fix...[/yellow]")
            fix_plan = await self.tester.fix_failures(self.goal, test_result.failures)

            # Create a fix task and execute it
            fix_task = TaskModel(
                id=f"fix-{self.cycles}",
                description=f"Fix test failures:\n{fix_plan}",
                files=[],
                dependencies=[],
            )
            self.all_tasks.append(fix_task)

            logger.info("Executing fix task", task_id=fix_task.id)
            with Status("[bold yellow]Applying fix...", console=console):
                fix_result = await WorkerAgent(task=fix_task, client=self.client, settings=self.settings).run()

            if isinstance(fix_result, Exception):
                logger.error("Fix task failed", error=str(fix_result))
                console.print(f"[red]Fix failed: {fix_result}[/red]")
            else:
                fix_task.status = "completed"
                fix_task.result = fix_result
                logger.info("Fix task completed")
                console.print("[green]Fix applied successfully[/green]")

            # Update goal for next cycle
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.all_tasks):
            return True
        return False
