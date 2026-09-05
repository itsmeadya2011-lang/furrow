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
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

log = get_logger(__name__)
console = Console()


class Orchestrator:
    """Coordinates the planning → execution → testing loop.

    The orchestrator runs an infinite (or bounded by ``max_cycles``) loop:

    1. **Plan** — the planner breaks the goal into parallelizable tasks.
    2. **Execute** — workers run concurrently, each with tool-use capability.
    3. **Test** — the tester runs the project's test suite and evaluates results.
    4. **Evaluate** — if tests pass, done; otherwise, the goal is refined
       to focus on fixing the failures and the loop repeats.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int | None = None,
        output_callback: "callable | None" = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.tests_passed: bool = False
        self.last_test_result: TestResult | None = None
        self._max_cycles_override = max_cycles
        self._output_callback = output_callback

    @property
    def max_cycles(self) -> int:
        if self._max_cycles_override is not None:
            return self._max_cycles_override
        return self.client.settings.max_cycles

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _output(self, message: Any) -> None:
        """Print to console and optionally forward to the output callback."""
        console.print(message)
        if not self._output_callback:
            return
        try:
            if isinstance(message, str):
                text = message
            else:
                # Render rich objects (Panel, Pretty, etc.) to plain text
                with console.capture() as cap:
                    console.print(message)
                text = cap.get()
            text = text.strip()
            if not text:
                return
            result = self._output_callback(text)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
        except Exception as e:
            log.error("Output callback failed", error=str(e))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._output(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._output(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

            if self._is_done():
                self._output("[bold green]Goal complete. Halting.[/bold green]")
                log.info("Goal complete", cycles=self.cycles)
                break

            # Enforce max_cycles (0 = unlimited)
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                self._output(
                    f"[yellow]Reached max_cycles ({self.max_cycles}). Stopping.[/yellow]"
                )
                log.warning("Max cycles reached", max_cycles=self.max_cycles)
                break

    async def _cycle(self) -> None:
        # 1 — Plan
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._output(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        log.info("Plan generated", cycle=self.cycles, tasks=len(plan.tasks))

        # Store tasks for tracking across cycles
        self.tasks = plan.tasks

        if not plan.tasks:
            self._output("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.tests_passed = True
            return

        # 2 — Execute in parallel (limited by max_parallel_tasks)
        max_parallel = self.client.settings.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel)

        async def _run_task(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_results = [_run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*worker_results, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._output(f"[red]Task {task.id} failed: {result}[/red]")
                log.error("Task failed", task_id=task.id, error=str(result), cycle=self.cycles)
            else:
                task.status = "completed"
                task.result = result
                self._output(f"[green]Task {task.id} completed[/green]")
                log.info("Task completed", task_id=task.id, cycle=self.cycles)

        # 3 — Test
        with Status("[bold yellow]Testing...", console=console):
            tester = TesterAgent(client=self.client)
            test_result = await tester.run(self.goal, plan.tasks)
        self.last_test_result = test_result

        if test_result.passed:
            self.tests_passed = True
            self._output(f"[green]Tests passed: {test_result.summary}[/green]")
            log.info("Tests passed", cycle=self.cycles)
        else:
            self.tests_passed = False
            self._output(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._output(f"  • {failure}")
            self._output("[yellow]Will attempt fix in next cycle.[/yellow]")
            log.warning("Tests failed", failures=test_result.failures, cycle=self.cycles)
            # Refine the goal to focus on the failing area
            failures_str = "\n".join(test_result.failures)
            self.goal = f"Fix failing tests:\n{failures_str}"

    def _is_done(self) -> bool:
        """Determine whether the orchestrator should stop.

        Done when:
        - No tasks were planned (goal likely already complete), OR
        - All tasks completed successfully AND tests passed.
        """
        if not self.tasks:
            return True

        all_completed = all(t.status == "completed" for t in self.tasks)
        any_failed = any(t.status == "failed" for t in self.tasks)

        if any_failed:
            # Some tasks failed — need another cycle
            return False

        if all_completed and self.tests_passed:
            return True

        return False

    def _get_tasks(self) -> list[Any]:
        """Return all tasks tracked by the orchestrator."""
        return self.tasks
