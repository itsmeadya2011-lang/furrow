from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult, WorkerResult
from furrow.llm import LLMClient
from furrow.config import settings as default_settings

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        emit: Callable[[str], Any] | None = None,
    ) -> None:
        self.goal = goal
        self.settings = default_settings
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.emit = emit
        self.all_tasks: list[TaskModel] = []
        self.cycles = 0

    async def _emit(self, message: str) -> None:
        if self.emit:
            await self.emit(message)
        else:
            console.print(message)

    async def run(self) -> None:
        banner = Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        if self.emit:
            await self.emit(f"Furrow — Goal: {self.goal}")
        else:
            console.print(banner)
        while True:
            self.cycles += 1
            await self._emit(f"\n=== Cycle {self.cycles} ===")
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                await self._emit(f"Max cycles ({self.settings.max_cycles}) reached. Halting.")
                break
            await self._cycle()
            if self._is_done():
                await self._emit("Goal complete. Halting.")
                break

    async def _cycle(self) -> None:
        plan: Plan | None = None
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        await self._emit("Plan: " + plan.rationale)
        await self._emit(str(Pretty(plan.model_dump())))

        if not plan.tasks:
            await self._emit("No tasks planned. Goal may be complete.")
            return

        # Track all tasks across cycles so completion can be detected
        self.all_tasks.extend(plan.tasks)

        # Execute tasks with concurrency limit
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def run_with_limit(task: TaskModel) -> WorkerResult:
            async with semaphore:
                worker = WorkerAgent(task=task, client=self.client)
                return await worker.run()

        await self._emit("Executing tasks in parallel...")
        results = await asyncio.gather(
            *(run_with_limit(task) for task in plan.tasks),
            return_exceptions=True,
        )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result.summary
                # Apply file changes written by the worker
                for file_edit in result.files:
                    await self.client.write_file(file_edit.path, file_edit.content)
                if result.issues:
                    for issue in result.issues:
                        await self._emit(f"  issue: {issue}")
                await self._emit(
                    f"Task {task.id} completed — {len(result.files)} file(s) written"
                )

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._emit(f"Tests passed: {test_result.summary}")
        else:
            await self._emit(f"Tests failed: {test_result.summary}")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("Will attempt fix in next cycle.")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        return completed >= len(self.all_tasks)
