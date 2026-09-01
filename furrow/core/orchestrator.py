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
from furrow.config import TaskModel
from furrow.llm import LLMClient

logger = structlog.get_logger(__name__)
console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.history: list[dict[str, Any]] = []
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)
        return self._semaphore

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("orchestrator.start", goal=self.goal)

        while True:
            self.cycles += 1

            # Enforce max_cycles (0 = infinite)
            max_cycles = self.client.settings.max_cycles
            if max_cycles > 0 and self.cycles > max_cycles:
                console.print(f"[yellow]Reached max_cycles ({max_cycles}). Halting.[/yellow]")
                logger.info("orchestrator.max_cycles_reached", cycles=self.cycles)
                break

            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("orchestrator.cycle_start", cycle=self.cycles)

            try:
                await self._cycle()
            except Exception as e:
                console.print(f"[red]Cycle {self.cycles} failed: {e}[/red]")
                logger.error("orchestrator.cycle_error", cycle=self.cycles, error=str(e))
                # Don't abort; try again next cycle
                continue

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("orchestrator.done", cycles=self.cycles)
                break

            # Brief pause between cycles
            await asyncio.sleep(0.1)

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("orchestrator.plan", tasks=len(plan.tasks), cycle=self.cycles)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("orchestrator.no_tasks", cycle=self.cycles)
            return

        # Track all tasks across cycles
        self.all_tasks.extend(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console) as status:
            worker_tasks = [
                asyncio.create_task(self._run_worker(task))
                for task in plan.tasks
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            # Update the task that's in all_tasks (same object reference from plan)
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("orchestrator.task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("orchestrator.task_completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("orchestrator.tests_passed", cycle=self.cycles)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("orchestrator.tests_failed", cycle=self.cycles, failures=test_result.failures)
            # Update goal for next cycle to fix failing tests
            fix_desc = (
                f"Fix failing tests from goal '{self.goal}':\n"
                + "\n".join(test_result.failures)
            )
            self.goal = fix_desc

        # Record cycle history
        self.history.append({
            "cycle": self.cycles,
            "tasks": [t.model_dump() for t in plan.tasks],
            "tests": test_result.model_dump(),
        })

    async def _run_worker(self, task: TaskModel) -> str:
        """Run a single worker with semaphore-based concurrency limiting."""
        async with self.semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _get_tasks(self) -> list[TaskModel]:
        """Return all tasks tracked across cycles."""
        return self.all_tasks

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True

        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        total = len(tasks)

        # If all tasks are resolved (completed or failed), we're done
        if completed + failed >= total:
            return True

        return False
