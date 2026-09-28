from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()

# Optional hook the orchestrator calls with plain-text progress messages.
# Used by the web UI to stream output to clients; ``None`` keeps console-only behavior.
ProgressCallback = Callable[[str], Awaitable[None]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.settings = self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        # Result of the most recent test run; drives the fix-loop in ``_is_done``.
        self.last_test_result: TestResult | None = None
        self._on_progress = on_progress
        self._semaphore: asyncio.Semaphore | None = None

    async def _progress(self, message: str, style: str = "") -> None:
        """Print a milestone to the console and forward it to the progress callback if any."""
        if style:
            console.print(message, style=style)
        else:
            console.print(message)
        if self._on_progress is not None:
            await self._on_progress(message)

    async def run(self) -> None:
        await self._progress(f"Furrow started — goal: {self.goal}", style="bold green")
        while True:
            self.cycles += 1
            await self._progress(f"═══ Cycle {self.cycles} ═══", style="bold cyan")

            # Respect the configured cycle cap (max_cycles == 0 means unlimited).
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                await self._progress(
                    f"Reached max_cycles ({self.settings.max_cycles}); stopping.", style="yellow"
                )
                break

            await self._cycle()

            if self._is_done():
                await self._progress("Goal complete. Halting.", style="bold green")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan: Plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._progress(f"Planned {len(plan.tasks)} task(s): {plan.rationale}", style="blue")

        if not plan.tasks:
            await self._progress("No tasks planned. Goal may be complete.", style="yellow")
            self.tasks = []
            return

        # Cap parallelism at the configured maximum (0/unset means no limit beyond task count).
        max_parallel = self.settings.max_parallel_tasks
        if max_parallel <= 0:
            max_parallel = len(plan.tasks)
        self._semaphore = asyncio.Semaphore(max_parallel)
        self.tasks = plan.tasks

        await self._progress(
            f"Executing {len(plan.tasks)} task(s) in parallel (capped at {max_parallel})...",
            style="yellow",
        )
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *(self._execute_task(task) for task in plan.tasks),
                return_exceptions=True,
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._progress(f"Task {task.id} failed: {result}", style="red")
            else:
                task.status = "completed"
                task.result = result
                await self._progress(f"Task {task.id} completed", style="green")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(
                client=self.client, workspace=self.settings.workspace
            ).run(self.goal, plan.tasks)
        self.last_test_result = test_result

        if test_result.passed:
            await self._progress(f"Tests passed: {test_result.summary}", style="green")
        else:
            console.print(
                Panel(
                    "\n".join(f"  • {f}" for f in test_result.failures) or test_result.summary,
                    title="Test failures",
                    border_style="red",
                )
            )
            await self._progress(
                f"Tests failed: {test_result.summary}; will retry fixes next cycle.",
                style="yellow",
            )
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_task(self, task: TaskModel) -> str:
        async with self._semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        """Return True only when there is no outstanding work and the latest tests passed."""
        # Nothing planned this cycle -> nothing left to do.
        if not self.tasks:
            return True
        # If the most recent tests failed, keep going to attempt a fix.
        if self.last_test_result is not None and not self.last_test_result.passed:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.tasks)
