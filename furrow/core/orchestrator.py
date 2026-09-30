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
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    """Runs the infinite parallel development loop.

    The orchestrator repeatedly:
      1. Plans tasks (via ``PlannerAgent``)
      2. Executes them in parallel (via ``WorkerAgent``)
      3. Tests the results (via ``TesterAgent``)
      4. Stops when tasks are complete, ``max_cycles`` is reached, or there are no tasks.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        *,
        max_cycles: int | None = None,
        custom_console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._current_plan: Plan | None = None
        # Tracks whether the last cycle's tests passed.  When tests fail the
        # loop should continue (even if all workers completed) so the planner
        # can fix the issues in the next cycle.
        self._tests_passed: bool = False
        # 0 / None means "no limit"; positive values cap the number of cycles.
        self.max_cycles = max_cycles if max_cycles is not None else self.client.settings.max_cycles
        # Allow callers (e.g. the web server) to inject a custom Console whose
        # output is captured and streamed back to the user.
        self.console: Console = custom_console if custom_console is not None else console

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def run(self) -> list[TaskModel]:
        """Execute the loop until the goal is complete or a stop condition is met.

        Returns the list of tasks from the final cycle.
        """
        self.console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            # Check the cycle limit *before* running so we respect the cap exactly.
            if self.max_cycles and self.cycles >= self.max_cycles:
                self.console.print(
                    f"[yellow]Reached max_cycles limit ({self.max_cycles}). Stopping.[/yellow]"
                )
                break

            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

        return self._get_tasks()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _cycle(self) -> None:
        # Plan -------------------------------------------------------
        with Status("[bold yellow]Planning...", console=self.console):
            plan = await self.planner.plan(self.goal)
        self._current_plan = plan
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Execute ----------------------------------------------------
        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            results = await self._execute_tasks(plan.tasks)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")

        # Test -------------------------------------------------------
        with Status("[bold yellow]Testing...", console=self.console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            failures_text = "\n".join(test_result.failures)
            self.goal = f"Fix failing tests:\n{failures_text}"
        self._tests_passed = test_result.passed

    async def _execute_tasks(self, tasks: list[TaskModel]) -> list[Any]:
        """Run worker agents in parallel, honouring ``max_parallel_tasks``."""
        max_parallel = self.client.settings.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel) if max_parallel > 0 else None

        async def _run(task: TaskModel) -> str:
            if semaphore is not None:
                async with semaphore:
                    return await WorkerAgent(task=task, client=self.client).run()
            return await WorkerAgent(task=task, client=self.client).run()

        return await asyncio.gather(
            *[_run(task) for task in tasks],
            return_exceptions=True,
        )

    def _get_tasks(self) -> list[TaskModel]:
        """Return the tasks from the most recent plan."""
        if self._current_plan is not None:
            return self._current_plan.tasks
        return []

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        failed = sum(1 for t in tasks if t.status == "failed")
        completed = sum(1 for t in tasks if t.status == "completed")
        if failed > 0:
            # Let the planner attempt to fix failures in the next cycle.
            return False
        if completed < len(tasks):
            # Some tasks haven't finished yet.
            return False
        # All workers completed — but we only stop if tests also passed.
        return self._tests_passed
