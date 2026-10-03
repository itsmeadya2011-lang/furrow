from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.llm import LLMClient

console = Console()

MAX_CONSECUTIVE_CYCLE_ERRORS = 3

# A cycle that neither completes its work nor turns the suite green is a stall:
# the planner keeps handing out tasks while the tests stay red, or every task
# fails again. MAX_CONSECUTIVE_STALLED_CYCLES of those in a row means the loop is
# burning API calls for nothing, so it stops and says why. This is the only
# backstop when ``max_cycles`` is 0 (unlimited, the default), and it mirrors
# MAX_CONSECUTIVE_CYCLE_ERRORS deliberately: both bounds are 3, both are
# per-run counters reset by genuine progress.
MAX_CONSECUTIVE_STALLED_CYCLES = 3


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        *,
        console: Console | None = None,
    ) -> None:
        self.original_goal: str = goal
        # Kept for backwards compatibility; never overwritten with test failures.
        self.goal: str = goal
        self.feedback: str | None = None
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        # Set when the planner returns a plan with no tasks: its way of saying the
        # goal is satisfied and there is nothing left to do.
        self._no_work_remaining = False
        # Consecutive cycles that moved nothing forward; reset by any real progress.
        self.consecutive_stalled_cycles = 0
        self.console = console

    # ------------------------------------------------------------------ console

    def _console(self) -> Console:
        """Return the injected console, falling back to the module-level global."""
        if self.console is not None:
            return self.console
        return console

    def _max_cycles(self) -> int:
        return int(getattr(self.client.settings, "max_cycles", 0) or 0)

    def _max_parallel(self) -> int:
        return max(1, int(getattr(self.client.settings, "max_parallel_tasks", 5) or 1))

    # --------------------------------------------------------------------- run

    async def run(self) -> None:
        out = self._console()
        out.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))

        max_cycles = self._max_cycles()
        consecutive_errors = 0

        while True:
            if max_cycles and self.cycles >= max_cycles:
                out.print(
                    f"[bold yellow]Reached cycle limit ({max_cycles}). "
                    "Stopping without declaring the goal complete.[/bold yellow]"
                )
                break

            self.cycles += 1
            out.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            try:
                made_progress = await self._cycle()
            except Exception as exc:  # noqa: BLE001 - a failed cycle must not kill the loop
                consecutive_errors += 1
                out.print(f"[bold red]Cycle {self.cycles} errored: {exc}[/bold red]")
                if consecutive_errors >= MAX_CONSECUTIVE_CYCLE_ERRORS:
                    out.print(
                        "[bold red]Aborting: "
                        f"{consecutive_errors} consecutive cycle errors.[/bold red]"
                    )
                    break
                continue

            consecutive_errors = 0

            if self._is_done():
                out.print(self._completion_message())
                break

            # Every cycle that does not finish the goal either resets or
            # increments the stall counter, so it always reaches the bound.
            if made_progress:
                self.consecutive_stalled_cycles = 0
                continue

            self.consecutive_stalled_cycles += 1
            if self.consecutive_stalled_cycles >= MAX_CONSECUTIVE_STALLED_CYCLES:
                out.print(
                    "[bold red]Aborting: no progress in the last "
                    f"{self.consecutive_stalled_cycles} consecutive cycles."
                    "[/bold red]"
                )
                out.print(
                    "[yellow]The goal is NOT verified. The outstanding failures are "
                    "listed above; fix them, or raise "
                    "MAX_CONSECUTIVE_STALLED_CYCLES for a longer retry budget."
                    "[/yellow]"
                )
                break

    # ------------------------------------------------------------------- cycle

    async def _cycle(self) -> bool:
        """Run one plan/execute/test cycle; return whether it made real progress."""
        out = self._console()

        with Status("[bold yellow]Planning...", console=out):
            plan = await self.planner.plan(self._goal_input())
        out.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            # An empty plan is the planner saying the goal is already satisfied:
            # there is nothing left to do, so asking again would only burn calls.
            self._no_work_remaining = True
            out.print(
                "[yellow]Planner reported no further work: "
                "nothing left to do.[/yellow]"
            )
            return False

        self.tasks.extend(plan.tasks)

        semaphore = asyncio.Semaphore(self._max_parallel())
        outcomes: list[tuple[TaskModel, Any]] = []

        with Status("[bold yellow]Executing tasks in parallel...", console=out):
            for stage in self._stage_tasks(plan.tasks):
                results = await asyncio.gather(
                    *(self._run_task(task, semaphore) for task in stage),
                    return_exceptions=True,
                )
                outcomes.extend(zip(stage, results))

        for task, result in outcomes:
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                out.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                out.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=out):
            test_result = await TesterAgent(client=self.client).run(
                self.original_goal, plan.tasks
            )

        if test_result.passed:
            self.feedback = None
            out.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.feedback = "\n".join(test_result.failures)
            out.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                out.print(f"  • {failure}")
            out.print("[yellow]Will attempt fix in next cycle.[/yellow]")

        # Only a green suite over wholly completed tasks is progress. Anything
        # else (red tests, a failed task) is a stall the loop must keep counting.
        return test_result.passed and all(task.status == "completed" for task in plan.tasks)

    def _goal_input(self) -> str:
        """The original goal plus any test feedback, for the planner."""
        if self.feedback:
            return f"{self.original_goal}\n\nFix failing tests:\n{self.feedback}"
        return self.original_goal

    # ---------------------------------------------------------------- scheduling

    @staticmethod
    def _stage_tasks(tasks: list[TaskModel]) -> list[list[TaskModel]]:
        """Group tasks into dependency stages, running each stage in order."""
        plan_ids = {task.id for task in tasks}
        scheduled: set[str] = set()
        pending = list(tasks)
        stages: list[list[TaskModel]] = []

        while pending:
            stage = [
                task
                for task in pending
                if all(
                    dep not in plan_ids or dep in scheduled
                    for dep in (task.dependencies or [])
                )
            ]
            if not stage:
                # Dependency cycle (or unsatisfiable deps): flush the rest as one batch.
                stages.append(pending)
                break
            stages.append(stage)
            scheduled.update(task.id for task in stage)
            stage_ids = {task.id for task in stage}
            pending = [task for task in pending if task.id not in stage_ids]

        return stages

    async def _run_task(self, task: TaskModel, semaphore: asyncio.Semaphore) -> Any:
        async with semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    # --------------------------------------------------------------------- state

    def _is_done(self) -> bool:
        """Whether the loop may halt and call the goal complete.

        Task statuses alone are not enough: a cycle can mark every task
        ``completed`` while the tester still reports failures, and treating that
        as done is a false pass. The suite must actually be green.
        """
        # The planner reporting no further work is its own completion signal,
        # even when earlier tasks failed or the suite stayed red.
        if self._no_work_remaining:
            return True
        tasks = self._get_tasks()
        if not tasks:
            return False
        if any(task.status == "failed" for task in tasks):
            return False
        if not all(task.status == "completed" for task in tasks):
            return False
        # All tasks completed but the tests are red: not done.
        return not self.feedback

    def _completion_message(self) -> str:
        """The line ``run()`` prints when it halts; never claims a false pass."""
        if self._no_work_remaining:
            if self.feedback or any(task.status == "failed" for task in self._get_tasks()):
                return (
                    "[bold yellow]Planner reported no further work, but the goal is "
                    "NOT verified: tests or tasks are still failing. The outstanding "
                    "failures are listed above.[/bold yellow]"
                )
            return "[bold green]Planner found no further work. Goal complete. Halting.[/bold green]"
        return "[bold green]Goal complete. Halting.[/bold green]"

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
