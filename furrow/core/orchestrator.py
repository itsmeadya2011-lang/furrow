from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
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
        on_event: Callable[[str], None] | Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient()
        self.settings = settings or Settings()
        self.on_event = on_event
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            await self._emit(f"\n═══ Cycle {self.cycles} ═══", "bold cyan")
            await self._cycle()
            if self._is_done():
                await self._emit("Goal complete. Halting.", "bold green")
                break
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                await self._emit(f"Max cycles ({self.settings.max_cycles}) reached. Halting.", "bold yellow")
                break

    async def _emit(self, message: str, style: str | None = None) -> None:
        if style:
            console.print(f"[{style}]{message}[/{style}]")
        else:
            console.print(message)
        if self.on_event is None:
            return
        try:
            if asyncio.iscoroutinefunction(self.on_event):
                await self.on_event(message)
            else:
                result = self.on_event(message)
                if inspect.isawaitable(result):
                    await result
        except Exception as exc:
            console.print(f"[dim]Event callback error: {exc}[/dim]")

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit(f"Plan complete: {len(plan.tasks)} task(s)", "bold blue")

        if not plan.tasks:
            await self._emit("No tasks planned. Goal may be complete.", "yellow")
            return

        self.tasks = plan.tasks

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            for task in self.tasks:
                task.status = "in_progress"
                await self._emit(f"Task {task.id} started: {task.description}", "cyan")
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in self.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(self.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"Task {task.id} failed: {result}", "red")
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"Task {task.id} completed", "green")

        await self._emit("Running tests...", "bold yellow")
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        self.last_test_result = test_result

        if test_result.passed:
            await self._emit(f"Tests passed: {test_result.summary}", "green")
        else:
            await self._emit(f"Tests failed: {test_result.summary}", "red")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}", "red")
            await self._emit("Will attempt fix in next cycle.", "yellow")
            self.goal = self.original_goal + "\n\nRecent failures to address:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        for task in tasks:
            if task.status in ("pending", "in_progress"):
                return False
        if self.last_test_result is not None and not self.last_test_result.passed:
            return False
        return True

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
