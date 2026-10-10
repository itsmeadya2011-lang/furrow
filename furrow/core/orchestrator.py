"""Orchestrator for Furrow runs.

Coordinates the planning, execution, and testing phases of a run,
persisting state between cycles via the StateManager.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult
from furrow.core.state import RunState, StateManager, TaskState
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger(__name__)


class Orchestrator:
    """Coordinates the plan-execute-test cycle for a Furrow run."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        state_manager: StateManager | None = None,
    ) -> None:
        """Initialize the Orchestrator.

        Args:
            goal: The goal to accomplish.
            client: Optional LLM client; created on demand if omitted.
            settings: Optional settings; defaults are used if omitted.
            state_manager: Optional state manager; created on demand if omitted.
        """
        self.goal = goal
        self.client = client or LLMClient()
        self.settings = settings or Settings()
        self.state_manager = state_manager or StateManager()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        self._test_result: TestResult | None = None
        self._history: list[str] = []

        # Attempt to load existing state
        self._load_state()

    def _load_state(self) -> None:
        """Load existing state from StateManager if available."""
        state = self.state_manager.load()
        if state:
            self.cycles = state.cycles
            # Convert TaskState back to TaskModel for internal use
            self._tasks = [
                TaskModel(
                    id=ts.id,
                    description=ts.description,
                    files=ts.files,
                    dependencies=[],  # TaskState doesn't have dependencies
                    status=ts.status,
                    result=ts.result,
                )
                for ts in state.tasks
            ]
            # If loaded state has a different goal, keep the current goal but log
            if state.goal != self.goal:
                logger.warning("Loaded state goal differs from current goal", loaded=state.goal, current=self.goal)
            logger.info("Resumed from saved state", cycles=self.cycles, tasks=len(self._tasks))

    def _save_state(self) -> None:
        """Save current state via StateManager."""
        now = datetime.now().isoformat()
        self._history.append(f"Cycle {self.cycles} completed at {now}")
        run_state = RunState(
            goal=self.goal,
            cycles=self.cycles,
            tasks=[
                TaskState(
                    id=t.id,
                    description=t.description,
                    files=t.files,
                    status=t.status,
                    result=t.result,
                )
                for t in self._tasks
            ],
            history=self._history,
            created_at=now,
            updated_at=now,
        )
        self.state_manager.save(run_state)

    def _get_tasks(self) -> list[TaskModel]:
        """Return the current tasks."""
        return self._tasks

    def _is_done(self) -> bool:
        """Check if the run is complete.

        Returns:
            False if no tasks planned yet (keep cycling).
            False if the latest test run failed (need fix cycle).
            False if any task failed (need fix cycle).
            True if all tasks completed.
        """
        if not self._tasks:
            logger.debug("No tasks planned yet, continuing")
            return False

        if self._test_result is not None and not self._test_result.passed:
            logger.debug("Latest test run failed, continuing for fix cycle")
            return False

        failed = any(t.status == "failed" for t in self._tasks)
        if failed:
            logger.debug("Some tasks failed, continuing for fix cycle")
            return False

        completed = all(t.status == "completed" for t in self._tasks)
        if completed:
            logger.info("All tasks completed")
            return True

        logger.debug("Tasks still in progress", completed=sum(1 for t in self._tasks if t.status == "completed"), total=len(self._tasks))
        return False

    async def run(self) -> None:
        """Run the plan-execute-test loop until the goal is complete."""
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("Starting run", goal=self.goal, max_cycles=self.settings.max_cycles, max_parallel=self.settings.max_parallel_tasks)

        while True:
            # Check max_cycles enforcement
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                logger.warning("Max cycles reached, stopping", cycles=self.cycles, max_cycles=self.settings.max_cycles)
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Stopping.[/yellow]")
                break

            self.cycles += 1
            logger.info("Starting cycle", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            await self._cycle()

            # Save state after each cycle
            self._save_state()

            if self._is_done():
                logger.info("Goal complete, halting", cycles=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        """Run a single plan, execute, and test cycle."""
        logger.debug("Planning phase started", cycle=self.cycles)
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        logger.info("Plan received", tasks=len(plan.tasks), rationale=plan.rationale)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.warning("No tasks planned")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Update internal task list from plan
        self._tasks = plan.tasks

        logger.debug("Execution phase started", cycle=self.cycles, task_count=len(plan.tasks))
        with Status("[bold yellow]Executing tasks in parallel...", console=console):

            async def run_with_semaphore(task: TaskModel) -> Any:
                """Run a single worker task, bounded by the concurrency semaphore."""
                async with self._semaphore:
                    return await WorkerAgent(task=task, client=self.client).run()

            tasks_coros = [run_with_semaphore(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks_coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
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

        logger.debug("Testing phase started", cycle=self.cycles)
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self._test_result = test_result

        if test_result.passed:
            logger.info("Tests passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("Tests failed", failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
