from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, settings
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int | None = None,
        max_parallel_tasks: int | None = None,
        output_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.max_cycles = max_cycles if max_cycles is not None else settings.max_cycles
        self.max_parallel_tasks = (
            max_parallel_tasks
            if max_parallel_tasks is not None
            else settings.max_parallel_tasks
        )
        self.output_callback = output_callback

    async def _emit(self, message: str) -> None:
        console.print(message)
        if self.output_callback is not None:
            await self.output_callback(message)

    async def run(self) -> None:
        logger.info("orchestrator_started", goal=self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        await self._emit(f"Furrow\nGoal: {self.goal}")
        while True:
            self.cycles += 1
            logger.info("cycle_started", cycle=self.cycles)
            await self._emit(f"═══ Cycle {self.cycles} ═══")
            await self._cycle()
            if self._is_done():
                logger.info("orchestrator_done")
                await self._emit("Goal complete. Halting.")
                break
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                logger.info("max_cycles_reached", max_cycles=self.max_cycles)
                await self._emit(
                    f"Reached max_cycles limit ({self.max_cycles}). Halting."
                )
                break

    async def _cycle(self) -> None:
        await self._emit("Planning...")
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        await self._emit("Planning complete.")
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit(f"Plan: {plan.model_dump()}")

        logger.info("planning_completed", task_count=len(plan.tasks))

        self.tasks = plan.tasks

        if not plan.tasks:
            logger.warning("no_tasks_planned")
            await self._emit("No tasks planned. Goal may be complete.")
            return

        semaphore = asyncio.Semaphore(self.max_parallel_tasks)

        async def _run_task(task: TaskModel) -> Any:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        await self._emit("Executing tasks in parallel...")
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            workers = [_run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*workers, return_exceptions=True)
        await self._emit("Execution complete.")

        completed = 0
        failed = 0
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                failed += 1
                logger.warning("task_failed", task_id=task.id, error=str(result))
                await self._emit(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                completed += 1
                await self._emit(f"Task {task.id} completed")

        logger.info("tasks_executed", completed=completed, failed=failed)

        await self._emit("Testing...")
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        await self._emit("Testing complete.")

        logger.info("tests_completed", passed=test_result.passed)

        if test_result.passed:
            await self._emit(f"Tests passed: {test_result.summary}")
        else:
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            await self._emit(f"Tests failed: {test_result.summary}")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            failure_details = "\n".join(f"  • {f}" for f in test_result.failures)
            self.goal = (
                f"Original goal: {self.original_goal}\n\n"
                f"Fix failing tests and address the following failures:\n"
                f"{failure_details}"
            )
            await self._emit("Will attempt fix in next cycle.")

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
