from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Coroutine

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


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_output: Callable[[str], None] | Callable[[str], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.completed_tasks: list[TaskModel] = []
        self.failed_tasks: list[TaskModel] = []
        self._on_output = on_output

    async def _emit(self, message: str) -> None:
        if self._on_output is None:
            return
        result = self._on_output(message)
        if asyncio.iscoroutine(result):
            await result

    async def run(self) -> None:
        await self._emit(f"Furrow starting. Goal: {self.goal}")
        while True:
            self.cycles += 1
            await self._emit(f"\n=== Cycle {self.cycles} ===")
            await self._cycle()
            if self._is_done():
                await self._emit("Goal complete. Halting.")
                break

    async def _cycle(self) -> None:
        await self._emit("Planning...")
        plan = await self.planner.plan(self.goal)
        await self._emit(f"Plan rationale: {plan.rationale}")

        if not plan.tasks:
            await self._emit("No tasks planned. Goal may be complete.")
            return

        # Merge new tasks with existing ones, avoiding duplicates by id
        existing_ids = {t.id for t in self.tasks}
        for task in plan.tasks:
            if task.id not in existing_ids:
                self.tasks.append(task)

        # Determine which tasks still need to run
        pending = [t for t in self.tasks if t.status == "pending"]
        if not pending:
            await self._emit("All tasks already processed.")
            return

        await self._emit(f"Executing {len(pending)} task(s) in parallel...")

        sem = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def run_with_limit(task: TaskModel) -> TaskModel:
            async with sem:
                try:
                    result = await WorkerAgent(task=task, client=self.client).run()
                    task.status = "completed"
                    task.result = result
                    self.completed_tasks.append(task)
                    await self._emit(f"[green]Task {task.id} completed[/green]")
                    return task
                except Exception as e:
                    task.status = "failed"
                    task.result = str(e)
                    self.failed_tasks.append(task)
                    await self._emit(f"[red]Task {task.id} failed: {e}[/red]")
                    return task

        results = await asyncio.gather(*(run_with_limit(t) for t in pending))

        await self._emit("Testing...")
        test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("Will attempt fix in next cycle.")
            # Reset failed tasks to pending so they get retried
            for task in self.failed_tasks:
                task.status = "pending"
            self.failed_tasks.clear()
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if self.failed_tasks:
            return False
        if self.client.settings.max_cycles and self.cycles >= self.client.settings.max_cycles:
            return True
        pending = sum(1 for t in self.tasks if t.status == "pending")
        if pending > 0:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        if completed >= len(self.tasks) and self.tasks:
            return True
        return False