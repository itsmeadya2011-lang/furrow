from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

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


async def _noop(_msg: str) -> None:
    pass


class Orchestrator:
    """Coordinates the plan -> execute -> test loop until the goal is complete."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        *,
        output: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._last_tasks: list[TaskModel] | None = None
        self.last_test_result: TestResult | None = None
        self.output: Callable[[str], Awaitable[None]] = output or _noop

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            self.cycles += 1
            await self._log(f"═══ Cycle {self.cycles} ═══", style="bold cyan")
            await self._cycle()

            max_cycles = self.client.settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                await self._log(f"Reached max_cycles ({max_cycles}). Stopping.", style="yellow")
                break

            if self._is_done():
                await self._log("Goal complete. Halting.", style="bold green")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._log(f"Plan rationale: {plan.rationale}")

        if not plan.tasks:
            await self._log("No tasks planned. Goal may be complete.", style="yellow")
            self._last_tasks = []
            self.last_test_result = TestResult(
                passed=True, summary="No tasks remain; goal is complete.", failures=[]
            )
            return

        max_parallel = max(1, self.client.settings.max_parallel_tasks)
        sem = asyncio.Semaphore(max_parallel)
        results = await asyncio.gather(
            *(self._run_task(sem, task) for task in plan.tasks),
            return_exceptions=True,
        )

        self._last_tasks = plan.tasks
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._log(f"Task {task.id} failed: {result}", style="red")
            else:
                task.status = "completed"
                task.result = result
                await self._log(f"Task {task.id} completed", style="green")

        with Status("[bold yellow]Testing...", console=console):
            self.last_test_result = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )

        if self.last_test_result.passed:
            await self._log(f"Tests passed: {self.last_test_result.summary}", style="green")
        else:
            await self._log(f"Tests failed: {self.last_test_result.summary}", style="red")
            for failure in self.last_test_result.failures:
                await self._log(f"  - {failure}")
            await self._log("Will attempt fix in next cycle.", style="yellow")
            self.goal = "Fix failing tests:\n" + "\n".join(self.last_test_result.failures)

    async def _run_task(self, sem: asyncio.Semaphore, task: TaskModel) -> str:
        async with sem:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        # Done only when the most recent cycle's tasks all completed and the
        # test suite passed. Otherwise the loop continues (bounded by max_cycles).
        if self._last_tasks is None or self.last_test_result is None:
            return False
        if any(task.status != "completed" for task in self._last_tasks):
            return False
        return bool(self.last_test_result.passed)

    async def _log(self, msg: str, style: str = "") -> None:
        if style:
            console.print(f"[{style}]{msg}[/{style}]")
        else:
            console.print(msg)
        await self.output(msg + "\n")
