from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_message: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.on_message = on_message
        self.max_cycles = self.settings.max_cycles
        self.max_parallel_tasks = self.settings.max_parallel_tasks
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._last_test_result: TestResult | None = None
        self._semaphore = asyncio.Semaphore(self.max_parallel_tasks)

    async def _log(self, message: str, style: str = "") -> None:
        console.print(message, style=style)
        if self.on_message is not None:
            await self.on_message(message)

    async def _log_panel(self, title: str, content: str) -> None:
        console.print(Panel(content, title=title))
        if self.on_message is not None:
            await self.on_message(content)

    async def run(self) -> None:
        await self._log_panel("Furrow", f"Goal: {self.goal}")
        while True:
            self.cycles += 1
            await self._log(f"Cycle {self.cycles}", "bold cyan")
            await self._cycle()
            if self._is_done():
                await self._log("Goal complete. Halting.", "bold green")
                break
            if self.max_cycles and self.cycles >= self.max_cycles:
                await self._log("Max cycles reached. Stopping.", "bold yellow")
                break

    async def _cycle(self) -> None:
        await self._log("Planning...", "bold yellow")
        with Status("[bold yellow]Planning...", console=console):
            plan: Plan = await self.planner.plan(self.goal)
        self._tasks = plan.tasks
        plan_json = json.dumps(plan.model_dump(), indent=2)
        await self._log_panel("Plan", plan_json)

        if not plan.tasks:
            await self._log("No tasks planned. Goal may be complete.", "yellow")
            return

        await self._log("Executing tasks in parallel...", "bold yellow")
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *(self._run_task(task) for task in plan.tasks),
                return_exceptions=True,
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._log(f"Task {task.id} failed: {result}", "red")
            else:
                task.status = "completed"
                task.result = result
                await self._log(f"Task {task.id} completed", "green")

        await self._log("Testing...", "bold yellow")
        with Status("[bold yellow]Testing...", console=console):
            self._last_test_result = await TesterAgent(
                client=self.client, settings=self.settings
            ).run(self.goal, plan.tasks)

        test_result = self._last_test_result
        if test_result.passed:
            await self._log(f"Tests passed: {test_result.summary}", "green")
        else:
            await self._log(f"Tests failed: {test_result.summary}", "red")
            for failure in test_result.failures:
                await self._log(f"  - {failure}")
            await self._log("Will attempt fix in next cycle.", "yellow")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _run_task(self, task: TaskModel) -> str:
        async with self._semaphore:
            return await WorkerAgent(task=task, client=self.client, settings=self.settings).run()

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if any(t.status != "completed" for t in tasks):
            return False
        if self._last_test_result is None:
            return True
        return self._last_test_result.passed

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks
