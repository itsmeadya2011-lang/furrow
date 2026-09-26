from __future__ import annotations

import asyncio
import time

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
from furrow.state import FurrowState

console = Console()
logger = structlog.get_logger(__name__)


class Orchestrator:
    """Runs the infinite development loop: plan -> execute -> test -> repeat."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        custom_settings: Settings | None = None,
    ) -> None:
        self.settings: Settings = custom_settings or settings
        self.goal = goal
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.current_plan: Plan | None = None
        self.last_test_result: TestResult | None = None
        self.last_test_passed: bool = False
        self._semaphore: asyncio.Semaphore | None = None

        # Resume from persisted state if available
        persisted = FurrowState.load(self.settings.state_file)
        if persisted is not None:
            self.goal = persisted.goal
            self.cycles = persisted.cycles
            self.all_tasks = list(persisted.all_tasks)
            self.last_test_result = persisted.last_test_result
            self.last_test_passed = (
                persisted.last_test_result.passed
                if persisted.last_test_result is not None
                else False
            )
            logger.info(
                "resumed from persisted state",
                cycles=self.cycles,
                tasks=len(self.all_tasks),
            )

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Lazily create the parallelism-limiting semaphore."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        return self._semaphore

    async def run(self) -> None:
        """Run the infinite development loop until the goal is complete."""
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"
            )
        )
        logger.info(
            "starting orchestrator",
            goal=self.goal,
            max_cycles=self.settings.max_cycles,
            max_parallel_tasks=self.settings.max_parallel_tasks,
        )

        try:
            while True:
                # Enforce max_cycles (0 = unlimited)
                if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                    console.print(
                        f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Stopping.[/yellow]"
                    )
                    logger.info("reached max_cycles", cycles=self.cycles)
                    break

                self.cycles += 1
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                await self._cycle()

                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    logger.info("goal complete", cycles=self.cycles)
                    break
        except KeyboardInterrupt:
            console.print("\n[yellow]Furrow stopped by user.[/yellow]")
            logger.info("orchestrator stopped by user")
        finally:
            self._save_state()

    async def _cycle(self) -> None:
        """Execute one planning -> execution -> testing cycle."""
        # 1. Plan
        with Status("[bold yellow]Planning...", console=console):
            try:
                plan = await self.planner.plan(self.goal)
            except Exception as e:
                logger.exception("planning failed", error=str(e))
                console.print(f"[red]Planning failed: {e}[/red]")
                self.current_plan = Plan(tasks=[], rationale=f"Planning failed: {e}")
                return

        console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )
        logger.info("plan created", tasks=len(plan.tasks), rationale=plan.rationale)
        self.current_plan = plan

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge planned tasks into the master task list
        self._merge_tasks(plan.tasks)

        # 2. Execute tasks respecting dependencies and parallelism
        await self._execute_tasks(plan.tasks)

        # 3. Test
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )

        self.last_test_result = test_result
        self.last_test_passed = test_result.passed

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  * {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        # Persist state after each cycle
        self._save_state()

    def _merge_tasks(self, tasks: list[TaskModel]) -> None:
        """Merge newly planned tasks into the master task list."""
        for task in tasks:
            existing = next((t for t in self.all_tasks if t.id == task.id), None)
            if existing is not None:
                existing.description = task.description
                existing.files = task.files
                existing.dependencies = task.dependencies
                existing.status = "pending"
                existing.result = None
            else:
                self.all_tasks.append(task)

    async def _execute_tasks(self, tasks: list[TaskModel]) -> None:
        """Execute tasks respecting dependencies with a parallelism limit."""
        completed_ids: set[str] = set()
        failed_ids: set[str] = set()

        pending = [t for t in tasks if t.status == "pending"]
        cycle_completed = 0
        cycle_failed = 0

        while pending:
            # Find tasks whose dependencies are all satisfied
            ready = [
                t
                for t in pending
                if all(
                    dep in completed_ids or dep in failed_ids
                    for dep in t.dependencies
                )
            ]

            if not ready:
                # Circular or unresolvable dependency
                for t in pending:
                    t.status = "failed"
                    t.result = f"Unresolvable dependency: {t.dependencies}"
                    failed_ids.add(t.id)
                    cycle_failed += 1
                    logger.error("unresolvable dependency", task_id=t.id)
                    console.print(
                        f"[red]Task {t.id} has unresolvable dependencies[/red]"
                    )
                break

            # Execute ready tasks in parallel (bounded by the semaphore)
            results = await asyncio.gather(
                *[self._run_worker(t) for t in ready],
                return_exceptions=True,
            )

            for task, result in zip(ready, results):
                pending.remove(task)
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    failed_ids.add(task.id)
                    cycle_failed += 1
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    logger.exception(
                        "worker task failed",
                        task_id=task.id,
                        error=str(result),
                    )
                else:
                    task.status = "completed"
                    task.result = result
                    completed_ids.add(task.id)
                    cycle_completed += 1
                    console.print(f"[green]Task {task.id} completed[/green]")

        logger.info(
            "cycle execution complete",
            completed=cycle_completed,
            failed=cycle_failed,
            total=len(tasks),
        )

    async def _run_worker(self, task: TaskModel) -> str:
        """Run a single worker task, bounded by the parallelism semaphore."""
        async with self.semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        """Check if the goal is complete.

        Returns True when the latest plan has no tasks (nothing to do) or
        when all tasks in the latest plan completed and tests passed.
        """
        if self.current_plan is None or not self.current_plan.tasks:
            return True

        tasks = self.current_plan.tasks
        all_completed = all(t.status == "completed" for t in tasks)
        return all_completed and self.last_test_passed

    def _get_tasks(self) -> list[TaskModel]:
        """Return all tasks tracked across all cycles."""
        return self.all_tasks

    def _save_state(self) -> None:
        """Persist current orchestrator state to disk."""
        state = FurrowState(
            goal=self.goal,
            cycles=self.cycles,
            all_tasks=self.all_tasks,
            last_test_result=self.last_test_result,
            updated_at=time.time(),
        )
        try:
            state.save(self.settings.state_file)
            logger.debug("state saved", path=str(self.settings.state_file))
        except OSError as e:
            logger.warning("failed to save state", error=str(e))
