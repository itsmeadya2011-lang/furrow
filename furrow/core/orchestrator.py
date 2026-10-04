from __future__ import annotations

import asyncio
from typing import cast

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow import config
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, settings
from furrow.llm import LLMClient

console = Console()

#: Consecutive failing cycles tolerated before the loop gives up. Bounds the
#: "fix the same failing tests forever" loop so ``run()`` always terminates.
MAX_CONSECUTIVE_FAILURES = 3

_SETTINGS_AT_IMPORT = settings


def _settings() -> Settings:
    """Return the live settings singleton.

    ``settings`` is re-exported on this module, so an override applied either
    here or on ``furrow.config`` is picked up at call time.
    """
    if settings is not _SETTINGS_AT_IMPORT:
        return settings
    return config.settings


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._failure_context: str | None = None
        self._consecutive_failures = 0
        self._no_tasks_planned = False
        #: Test verdict of the most recent cycle. ``None`` means "no verdict has
        #: been recorded yet", which is not by itself a failure.
        self._tests_passed: bool | None = None

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            max_cycles = _settings().max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print(
                    f"[bold yellow]Cycle budget exhausted after {self.cycles} cycle(s) "
                    f"(max_cycles={max_cycles}). Halting.[/bold yellow]"
                )
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self._no_tasks_planned:
                console.print("[yellow]Planner produced no tasks. Halting.[/yellow]")
                break
            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                console.print(
                    "[bold red]No progress after "
                    f"{self._consecutive_failures} consecutive failing cycles. Halting.[/bold red]"
                )
                break

    async def _cycle(self) -> None:
        self._no_tasks_planned = False
        # A verdict belongs to the cycle that produced it: drop the previous
        # one up front so ``_is_done()`` never reads a stale verdict.
        self._tests_passed = None
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self._planning_goal())
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._no_tasks_planned = True
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self._store_plan(plan)
        await self._execute_tasks()

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self._tasks)

        self._tests_passed = test_result.passed

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self._failure_context = "Fix failing tests:\n" + "\n".join(test_result.failures)

        self._track_progress(test_result.passed)

    def _store_plan(self, plan: Plan) -> None:
        """Retain the plan's tasks so progress survives across cycles."""
        self._tasks = list(plan.tasks)

    def _planning_goal(self) -> str:
        """Original goal, plus outstanding failure context; ``goal`` is never clobbered."""
        if self._failure_context:
            return f"{self.goal}\n\n{self._failure_context}"
        return self.goal

    def _track_progress(self, tests_passed: bool) -> None:
        """Reset or advance the consecutive-failure counter used for termination."""
        failed_tasks = [t.id for t in self._tasks if t.status == "failed"]
        if tests_passed and not failed_tasks:
            self._failure_context = None
            self._consecutive_failures = 0
            return
        if tests_passed and failed_tasks and not self._failure_context:
            self._failure_context = (
                "Repair or redo the following failed tasks:\n" + "\n".join(failed_tasks)
            )
        self._consecutive_failures += 1

    async def _execute_tasks(self) -> None:
        """Run the retained tasks, respecting dependencies and the parallelism cap."""
        cap = _settings().max_parallel_tasks
        completed = {t.id for t in self._tasks if t.status == "completed"}
        remaining = [t for t in self._tasks if t.status != "completed"]

        while remaining:
            ready = [i for i, task in enumerate(remaining) if self._deps_met(task, completed)]
            if not ready:
                # Missing or cyclic dependencies: relax them rather than deadlock.
                console.print(
                    "[yellow]Unsatisfiable or cyclic task dependencies; "
                    "running remaining tasks anyway.[/yellow]"
                )
                ready = list(range(len(remaining)))

            batch_idx = ready if cap <= 0 else ready[:cap]
            if cap > 0 and len(ready) > len(batch_idx):
                console.print(
                    f"[yellow]Deferring {len(ready) - len(batch_idx)} task(s) to respect "
                    f"max_parallel_tasks={cap}; they run in a later batch.[/yellow]"
                )
            batch = [remaining[i] for i in batch_idx]

            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                tasks = [WorkerAgent(task=task, client=self.client).run() for task in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            # ``batch`` and ``tasks`` are built 1:1 from each other and gather
            # returns one element per awaitable, so the lengths always agree.
            for task, result in zip(batch, results, strict=True):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    # ``cast`` rather than a wider isinstance: widening the check
                    # to BaseException would reclassify a CancelledError as a
                    # failed task, which is a behaviour change.
                    task.result = cast("str", result)
                    console.print(f"[green]Task {task.id} completed[/green]")
                if task.status == "completed":
                    completed.add(task.id)

            done = set(batch_idx)
            remaining = [t for i, t in enumerate(remaining) if i not in done]

    @staticmethod
    def _deps_met(task: TaskModel, completed: set[str]) -> bool:
        """True when every dependency of ``task`` is already completed."""
        return all(dep in completed for dep in task.dependencies)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        if any(t.status == "failed" for t in tasks):
            return False
        if not all(t.status == "completed" for t in tasks):
            return False
        # Completed tasks are not verified work: a cycle whose tests failed has
        # not achieved the goal yet, whatever the task statuses say, so the
        # repair cycle gets a chance to run.
        if self._tests_passed is False:
            return False
        return True

    def _get_tasks(self) -> list[TaskModel]:
        """Tasks of the most recent plan, retained between cycles."""
        return self._tasks
