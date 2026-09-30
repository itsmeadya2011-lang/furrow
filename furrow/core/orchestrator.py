from __future__ import annotations

import asyncio
from collections.abc import Callable

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

console = Console()
logger = structlog.get_logger("furrow.orchestrator")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        # Track tasks across cycles so _is_done() can observe their state.
        self.tasks: list[TaskModel] = []
        # 0 means infinite (no cap); otherwise stop after this many cycles.
        self.max_cycles: int = self.client.settings.max_cycles
        # Optional listener for streaming progress (e.g. from the web server).
        self.progress_callback = progress_callback
        self.logger = logger

    def _emit(self, status: str) -> None:
        """Notify progress listeners of a status update."""
        self.logger.info("progress", message=status)
        if self.progress_callback is not None:
            try:
                self.progress_callback(status)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("progress_callback_error", error=str(exc))

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit(f"Cycle {self.cycles} starting")
            await self._cycle()
            # Stop if all tasks completed (tests passing).
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._emit("Goal complete")
                break
            # Enforce the configured cycle cap (0 = infinite).
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                console.print(
                    f"[bold yellow]Reached max_cycles ({self.max_cycles}). Halting.[/bold yellow]"
                )
                self._emit(f"Reached max_cycles ({self.max_cycles}); halting")
                break

    async def _cycle(self) -> None:
        self.logger.info("cycle_start", cycle=self.cycles, goal=self.goal)

        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        # Persist planned tasks into instance state so their status survives
        # beyond this cycle (fixes the loop that never progressed).
        self.tasks = plan.tasks
        self.logger.info("cycle_planned", cycle=self.cycles, num_tasks=len(self.tasks))

        if not self.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._emit("No tasks planned")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            await asyncio.gather(
                *(self._execute_task(task) for task in self.tasks),
                return_exceptions=True,
            )

        # _execute_task updates each task's status/result in place.
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        self.logger.info(
            "cycle_tasks_executed", cycle=self.cycles, completed=completed, failed=failed
        )
        self._emit(f"Tasks executed: {completed} completed, {failed} failed")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        self.logger.info(
            "test_result",
            passed=test_result.passed,
            summary=test_result.summary,
            failures=test_result.failures,
        )
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._emit(f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self._emit(f"Tests failed: {test_result.summary}")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        self.logger.info("cycle_end", cycle=self.cycles, goal_complete=self._is_done())

    async def _execute_task(self, task: TaskModel) -> None:
        """Run a single worker task, retrying failed attempts up to 3 times.

        On retry the task description is augmented with the previous error so
        the worker receives adjusted instructions for the next attempt. The
        passed-in ``task`` is mutated in place (status/result) so the instance
        state tracked in ``self.tasks`` stays consistent.
        """
        max_retries = 3
        original_description = task.description
        last_error: str | None = None

        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                # Adjusted instructions for the retry: explain what failed.
                task.description = (
                    f"{original_description}\n\n"
                    f"[Retry attempt {attempt}/{max_retries}] "
                    f"The previous attempt raised an error:\n{last_error}\n"
                    "Please address this and retry the task."
                )

            worker = WorkerAgent(task=task, client=self.client)
            try:
                result = await worker.run()
            except Exception as exc:
                last_error = str(exc) or repr(exc)
                task.result = last_error
                if attempt < max_retries:
                    self.logger.warning(
                        "task_retry",
                        task_id=task.id,
                        attempt=attempt,
                        error=last_error,
                    )
                    self._emit(f"Task {task.id} failed (attempt {attempt}/{max_retries}); retrying")
                    continue
                task.status = "failed"
                self.logger.error("task_failed", task_id=task.id, error=last_error)
                self._emit(f"Task {task.id} failed after {max_retries} attempts")
                console.print(f"[red]Task {task.id} failed: {last_error}[/red]")
                return

            task.status = "completed"
            task.result = result
            self.logger.info("task_completed", task_id=task.id, attempt=attempt)
            self._emit(f"Task {task.id} completed")
            console.print(f"[green]Task {task.id} completed[/green]")
            return

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        # Return the task list tracked in instance state instead of an empty
        # list, otherwise _is_done() short-circuits to True every cycle.
        return self.tasks
