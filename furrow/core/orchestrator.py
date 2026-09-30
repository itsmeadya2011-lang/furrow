"""The Furrow orchestrator drives the plan/execute/test loop.

It owns the high-level lifecycle: plan a set of tasks, execute them in
dependency-aware waves (honouring ``max_parallel_tasks``), run the test suite,
and repeat until the goal is complete or we hit ``max_cycles``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

console = Console()


def _resolve_log_level(level_name: str) -> int:
    """Convert a textual logging level (e.g. "INFO") into a numeric level.

    ``Settings.log_level`` maps to the ``FURROW_LOG_LEVEL`` env var. Invalid or
    unknown values fall back to ``INFO`` rather than raising.
    """
    level = logging.getLevelName(level_name.upper())
    if isinstance(level, int) and level >= 0:
        return level
    return logging.INFO


def _build_logger(settings: Settings) -> structlog.BoundLogger:
    """Create and configure a structlog logger from ``settings``.

    The logger is configured with a stable set of processors (timestamps, level,
    stack-info and a human-readable console renderer) and filtered at
    ``settings.log_level`` via structlog's filtering bound logger so that
    ``FURROW_LOG_LEVEL`` is honoured.
    """
    level = _resolve_log_level(settings.log_level)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=False,
    )
    return structlog.get_logger("furrow")


class Orchestrator:
    """Coordinates planning, task execution and testing for a goal.

    Attributes:
        goal: The current goal string. When tests fail it is rewritten to ask
            for fixes, driving the next cycle.
        settings: Runtime configuration (max cycles, parallelism, log level...).
        all_tasks: The tasks from the most recent plan. Used by ``_is_done`` so
            that completion is computed against the *current* cycle's work.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        # Tasks produced by the most recent planning step. This is the state
        # used by ``_is_done`` / ``_get_tasks`` to decide whether to stop,
        # fixing the old bug where ``_get_tasks`` always returned ``[]``.
        self.all_tasks: list[TaskModel] = []
        self.logger = _build_logger(self.settings)

    async def run(self) -> None:
        """Run the plan/execute/test loop until completion or cycle limit."""
        self.logger.info(
            "run_started", goal=self.goal, max_cycles=self.settings.max_cycles,
            max_parallel_tasks=self.settings.max_parallel_tasks,
        )
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )

        while True:
            # Enforce the cycle budget (0 == unlimited).
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self.logger.info(
                    "loop_terminated", reason="max_cycles_reached",
                    cycles=self.cycles, max_cycles=self.settings.max_cycles,
                )
                console.print(
                    f"[yellow]Reached max_cycles "
                    f"({self.settings.max_cycles}). Stopping.[/yellow]"
                )
                break

            self.cycles += 1
            self.logger.info("cycle_start", cycle=self.cycles, goal=self.goal)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            await self._cycle()

            if self._is_done():
                self.logger.info("goal_complete", cycle=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        """Execute a single plan/execute/test pass."""
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)

        self.logger.info(
            "plan_created", cycle=self.cycles, task_count=len(plan.tasks),
            rationale=plan.rationale,
        )
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        # Refresh instance task state so ``_is_done`` reflects this cycle.
        self.all_tasks = plan.tasks

        if not plan.tasks:
            self.logger.info("no_tasks_planned", cycle=self.cycles)
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        await self._execute_tasks(plan.tasks)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(
                client=self.client, settings=self.settings
            ).run(self.goal, plan.tasks)

        self.logger.info(
            "test_result", passed=test_result.passed, summary=test_result.summary,
            failures=test_result.failures,
        )
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------
    def _compute_waves(self, tasks: list[TaskModel]) -> list[list[TaskModel]]:
        """Group ``tasks`` into dependency-ordered waves.

        Each wave contains tasks whose dependencies were all placed in earlier
        waves, so waves can be executed in order. The implementation is a pure
        topological layering (Kahn's algorithm by level) and is independent of
        any LLM, making it straightforward to unit test.

        Raises:
            ValueError: if a dependency is unknown or there is a cycle.
        """
        if not tasks:
            return []

        by_id = {task.id: task for task in tasks}
        for task in tasks:
            for dep in task.dependencies:
                if dep not in by_id:
                    raise ValueError(
                        f"Task '{task.id}' depends on unknown task '{dep}'"
                    )

        waves: list[list[TaskModel]] = []
        completed: set[str] = set()
        remaining = {task.id for task in tasks}
        while remaining:
            ready = [
                task
                for task in tasks
                if task.id in remaining
                and all(dep in completed for dep in task.dependencies)
            ]
            if not ready:
                unresolved = sorted(remaining)
                raise ValueError(
                    f"Unresolvable dependency cycle among: {unresolved}"
                )
            waves.append(ready)
            for task in ready:
                completed.add(task.id)
                remaining.discard(task.id)
        return waves

    async def _execute_tasks(self, tasks: list[TaskModel]) -> None:
        """Run ``tasks`` in dependency-aware waves.

        Waves run sequentially (a later wave only starts once the previous one
        has completed). Within a wave, tasks run concurrently but are bounded by
        a semaphore sized to ``settings.max_parallel_tasks``. Tasks whose
        dependencies failed in an earlier wave are skipped and marked failed
        rather than executed.
        """
        waves = self._compute_waves(tasks)
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        completed: set[str] = set()
        failed: set[str] = set()

        for wave in waves:
            runnable: list[TaskModel] = []
            for task in wave:
                if any(dep in failed for dep in task.dependencies):
                    task.status = "failed"
                    task.result = "Skipped: dependency failed"
                    failed.add(task.id)
                    self.logger.warning(
                        "task_blocked", task_id=task.id, reason="dependency_failed"
                    )
                    console.print(
                        f"[yellow]Task {task.id} skipped (dependency failed)[/yellow]"
                    )
                else:
                    runnable.append(task)

            if not runnable:
                continue

            self.logger.info(
                "wave_start", wave_number=len(completed) + len(failed),
                task_ids=[t.id for t in runnable],
            )

            coros = [self._run_task(task, semaphore) for task in runnable]
            results = await asyncio.gather(*coros, return_exceptions=True)

            for task, result in zip(runnable, results):
                if isinstance(result, BaseException):
                    task.status = "failed"
                    task.result = str(result)
                    failed.add(task.id)
                    self.logger.error(
                        "task_failed", task_id=task.id, error=str(result),
                    )
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    completed.add(task.id)
                    self.logger.info("task_completed", task_id=task.id)
                    console.print(f"[green]Task {task.id} completed[/green]")

    async def _run_task(
        self, task: TaskModel, semaphore: asyncio.Semaphore
    ) -> str:
        """Execute a single task via a worker, bounded by ``semaphore``.

        Split out from ``_execute_tasks`` so that tests can override this to run
        without an LLM.
        """
        async with semaphore:
            worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
            return await worker.run()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def _get_tasks(self) -> list[Any]:
        """Return the tasks from the most recent plan (instance state)."""
        return self.all_tasks

    def _is_done(self) -> bool:
        """Return ``True`` when every task in the current cycle is complete.

        Failure of any task means the cycle is *not* considered done: the loop
        continues so the next cycle can attempt a fix.
        """
        tasks = self._get_tasks()
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)
