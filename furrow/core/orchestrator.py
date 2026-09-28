from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_callback: Callable[[str], Any] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.output_callback = output_callback
        self.cycles = 0
        self.plan: Plan | None = None
        self.tasks: list[Any] = []
        self.last_test_result: TestResult | None = None

    async def _emit(self, msg: str, style: str = "") -> None:
        """Emit a message to the configured callback (plain text) or the console."""
        if self.output_callback is not None:
            result = self.output_callback(msg)
            if asyncio.iscoroutine(result):
                await result
        else:
            console.print(msg, style=style)

    async def run(self) -> None:
        if self.output_callback is not None:
            await self._emit(f"Furrow\nGoal: {self.goal}")
        else:
            console.print(
                Panel.fit(
                    f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"
                )
            )
        while True:
            self.cycles += 1
            await self._emit(f"\n═══ Cycle {self.cycles} ═══", style="bold cyan")
            await self._cycle()
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                await self._emit("Reached max_cycles limit. Halting.", style="yellow")
                break
            if self._is_done():
                await self._emit("Goal complete. Halting.", style="bold green")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        if self.output_callback is not None:
            await self._emit(json.dumps(plan.model_dump(), indent=2), style="blue")
        else:
            console.print(
                Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
            )

        self.plan = plan
        self.tasks = plan.tasks

        if not self.tasks:
            await self._emit("No tasks planned. Goal may be complete.", style="yellow")
            return

        max_parallel = self.client.settings.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_worker(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *(run_worker(task) for task in self.tasks), return_exceptions=True
            )

        for task, result in zip(self.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"Task {task.id} failed: {result}", style="red")
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"Task {task.id} completed", style="green")

        # Retry failed tasks once, prefixed with the failure context.
        failed = [t for t in self.tasks if t.status == "failed"]
        if failed:

            async def run_fix(task: TaskModel) -> tuple[TaskModel, str | Exception]:
                error = task.result or "unknown error"
                fix_description = (
                    "FIX instructions: the previous attempt failed with this error:\n"
                    f"{error}\n\nImplement this task correctly.\n\n"
                    f"Task: {task.description}"
                )
                retry_task = task.model_copy(update={"description": fix_description})
                async with semaphore:
                    try:
                        outcome: str | Exception = await WorkerAgent(
                            task=retry_task, client=self.client
                        ).run()
                    except Exception as exc:  # noqa: BLE001
                        outcome = exc
                return retry_task, outcome

            retry_results = await asyncio.gather(*(run_fix(t) for t in failed))
            for retry_task, outcome in retry_results:
                for task in self.tasks:
                    if task.id == retry_task.id:
                        task.description = retry_task.description
                        if isinstance(outcome, BaseException):
                            task.status = "failed"
                            task.result = str(outcome)
                            await self._emit(
                                f"Task {task.id} retry failed: {outcome}", style="red"
                            )
                        else:
                            task.status = "completed"
                            task.result = outcome
                            await self._emit(
                                f"Task {task.id} retry completed", style="green"
                            )
                        break

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)
        self.last_test_result = test_result

        if test_result.passed:
            await self._emit(f"Tests passed: {test_result.summary}", style="green")
        else:
            await self._emit(f"Tests failed: {test_result.summary}", style="red")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("Will attempt fix in next cycle.", style="yellow")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        failed = [t for t in tasks if t.status == "failed"]
        if failed:
            return False
        if not tasks:
            # No work left to do.
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        if completed < len(tasks):
            return False
        # All tasks completed with no failures. The loop is done only when the
        # latest test run passed; otherwise keep going to fix failing tests.
        if self.last_test_result is None:
            return False
        return self.last_test_result.passed

    def _get_tasks(self) -> list[Any]:
        return self.tasks

