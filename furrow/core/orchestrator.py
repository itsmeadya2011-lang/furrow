from __future__ import annotations

import asyncio
from collections.abc import Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    """Orchestrates the planning → execution → testing loop.

    Each cycle:
      1. Planner breaks the goal into parallelizable tasks.
      2. Workers execute tasks (bounded by ``max_parallel_tasks``).
      3. Tester runs the test suite and evaluates results.
      4. If tests fail, the next cycle targets the failures.
    The loop terminates when all tasks pass or ``max_cycles`` is reached.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int | None = None,
        max_parallel: int | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        from furrow.config import settings as global_settings

        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self._max_cycles: int = max_cycles or global_settings.max_cycles
        self._max_parallel: int = max_parallel or global_settings.max_parallel_tasks
        self._progress_callback = progress_callback

        # All tasks across all cycles (for completion tracking)
        self.all_tasks: list[TaskModel] = []

        # Track whether the current cycle's work is done
        self._last_test_result: TestResult | None = None

    # ------------------------------------------------------------------
    # Progress / logging
    # ------------------------------------------------------------------

    def _emit(self, message: str) -> None:
        """Send a progress update to the callback (if set) and console."""
        if self._progress_callback:
            self._progress_callback(message)

    def _print(self, message: str, style: str | None = None) -> None:
        if style:
            console.print(f"[{style}]{message}[/{style}]")
        else:
            console.print(message)
        self._emit(message)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        self._emit("Furrow started")

        while True:
            self.cycles += 1
            self._print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]", "bold cyan")
            self._emit(f"Cycle {self.cycles} starting")

            await self._cycle()

            if self._is_done():
                self._print("[bold green]Goal complete. Halting.[/bold green]", "bold green")
                self._emit("Goal complete. Halting.")
                break

            if self._max_cycles > 0 and self.cycles >= self._max_cycles:
                self._print(
                    f"\n[yellow]Max cycles ({self._max_cycles}) reached. Stopping.[/yellow]",
                    "yellow",
                )
                self._emit(f"Max cycles ({self._max_cycles}) reached. Stopping.")
                break

        self._emit("Furrow finished")
        console.print("\n[bold]Final task summary:[/bold]")
        for t in self.all_tasks:
            console.print(f"  [{t.status}] {t.id}: {t.description}")

    async def _cycle(self) -> None:
        # --- Planning ---
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._print(
            Panel(
                Pretty(plan.model_dump()),
                title=f"Plan (Cycle {self.cycles})",
                border_style="blue",
            )
        )
        self._emit(f"Plan created with {len(plan.tasks)} tasks")

        if not plan.tasks:
            self._print("[yellow]No tasks planned. Goal may be complete.[/yellow]", "yellow")
            self._emit("No tasks planned")
            return

        # --- Execution (parallel with semaphore) ---
        semaphore = asyncio.Semaphore(self._max_parallel)

        async def _bounded_run(task: TaskModel) -> TaskModel:
            async with semaphore:
                return await self._execute_task(task)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [_bounded_run(task) for task in plan.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._print(f"[red]Task {task.id} failed: {result}[/red]", "red")
                self._emit(f"Task {task.id} failed")
            else:
                task.status = "completed"
                task.result = result
                self._print(f"[green]Task {task.id} completed[/green]", "green")
                self._emit(f"Task {task.id} completed")

            # Track in all_tasks (replace if same ID, append otherwise)
            existing = next((t for t in self.all_tasks if t.id == task.id), None)
            if existing:
                existing.status = task.status
                existing.result = task.result
            else:
                self.all_tasks.append(task)

        # --- Testing ---
        with Status("[bold yellow]Testing...", console=console):
            test_result = await self.tester.run(self.goal, plan.tasks)
        self._last_test_result = test_result

        if test_result.passed:
            self._print(f"[green]Tests passed: {test_result.summary}[/green]", "green")
            self._emit("Tests passed")
        else:
            self._print(f"[red]Tests failed: {test_result.summary}[/red]", "red")
            for failure in test_result.failures:
                self._print(f"  • {failure}", "red")
            self._print("[yellow]Will attempt fix in next cycle.[/yellow]", "yellow")
            self._emit(f"Tests failed: {len(test_result.failures)} failures")
            # Reframe goal to focus on fixing failures
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_task(self, task: TaskModel) -> str:
        """Execute a single task via the WorkerAgent."""
        worker = WorkerAgent(task=task, client=self.client)
        return await worker.run()

    def _is_done(self) -> bool:
        """Check if all tasks in the current plan have completed or failed
        and tests pass."""
        current_plan_tasks = self.all_tasks

        # If no tasks have been run yet, continue
        if not current_plan_tasks:
            return False

        # Check if all tasks have a terminal status
        all_terminal = all(t.status in ("completed", "failed") for t in current_plan_tasks)

        # If any task failed, we're not done — we'll try again next cycle
        has_failures = any(t.status == "failed" for t in current_plan_tasks)

        # Tests must pass for completion
        tests_pass = self._last_test_result.passed if self._last_test_result else False

        if has_failures:
            return False
        return bool(all_terminal and tests_pass)
