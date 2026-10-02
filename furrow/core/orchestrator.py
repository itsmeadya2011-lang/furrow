from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger(__name__)


class Orchestrator:
    """Runs the Furrow development loop: plan -> execute -> test -> repeat."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings = default_settings,
    ) -> None:
        self.goal = goal
        self.settings = settings
        self.client = client or LLMClient(settings=settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.workspace = Path(self.settings.workspace)
        self._current_tasks: list[TaskModel] = []
        self._current_tests_passed = True

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("Starting Furrow orchestrator", goal=self.goal, max_cycles=self.settings.max_cycles)
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print("[yellow]Max cycles reached. Halting.[/yellow]")
                logger.info("Max cycles reached, halting", cycles=self.cycles)
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("Goal complete, halting")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("Plan created", tasks=len(plan.tasks), rationale=plan.rationale)

        self._current_tasks = plan.tasks
        self._current_tests_passed = True

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        results = await self._execute_tasks(plan.tasks)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self._current_tests_passed = test_result.passed

        self._print_test_result(test_result)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("Tests passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("Tests failed", failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[TaskModel]) -> list[Any]:
        """Execute tasks respecting dependencies and max_parallel_tasks limit.

        Tasks are executed in waves: only tasks whose dependencies are already
        completed or absent are eligible. Within a wave, up to max_parallel_tasks
        tasks run concurrently.
        """
        remaining = list(tasks)
        completed_ids: set[str] = set()
        processed_ids: set[str] = set()
        all_results: list[Any] = [None] * len(tasks)

        while remaining:
            # Find tasks whose dependencies are satisfied
            ready = [
                t for t in remaining
                if all(dep in completed_ids for dep in t.dependencies)
            ]

            if not ready:
                # Tasks remain but none are ready - they have unmet deps
                # This shouldn't normally happen if deps are valid
                logger.error("Deadlock: unresolvable task dependencies", remaining=[t.id for t in remaining])
                for task in remaining:
                    idx = tasks.index(task)
                    all_results[idx] = RuntimeError(
                        f"Unresolvable dependency for task {task.id}: "
                        f"deps={task.dependencies}, completed={completed_ids}"
                    )
                    task.status = "failed"
                    task.result = "Unresolvable dependencies"
                remaining.clear()
                break

            # Limit parallelism
            batch = ready[: self.settings.max_parallel_tasks]
            console.print(f"[dim]Executing {len(batch)} task(s) in parallel "
                          f"(total: {len(remaining)})[/dim]")

            coros = [WorkerAgent(task=task, client=self.client).run() for task in batch]
            results = await asyncio.gather(*coros, return_exceptions=True)

            for task, result in zip(batch, results):
                idx = tasks.index(task)
                all_results[idx] = result
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    processed_ids.add(task.id)
                    logger.error("Task failed", task_id=task.id, error=str(result))
                else:
                    completed_ids.add(task.id)
                    processed_ids.add(task.id)
                    task.status = "completed"
                    task.result = result
                    logger.info("Task completed", task_id=task.id)

            remaining = [t for t in remaining if t.id not in processed_ids]

        return all_results

    def _print_test_result(self, test_result: TestResult) -> None:
        table = Table(title="Test Results", border_style="magenta")
        table.add_column("Passed", style="green")
        table.add_column("Summary", overflow="fold")
        table.add_row(str(test_result.passed), test_result.summary or "(empty)")
        console.print(table)

    def _is_done(self) -> bool:
        """Determine if the goal is complete based on the current cycle's state.

        Returns True when:
        - No tasks were planned in the current cycle (planner says goal is done), OR
        - All tasks in the current cycle completed successfully AND tests passed.
        """
        if not self._current_tasks:
            return True
        all_completed = all(t.status == "completed" for t in self._current_tasks)
        return all_completed and self._current_tests_passed

    def _get_tasks(self) -> list[TaskModel]:
        """Return the tasks from the current cycle's plan."""
        return self._current_tasks
