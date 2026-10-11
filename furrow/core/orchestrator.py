from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Any | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []
        self._semaphore: asyncio.Semaphore | None = None
        self.on_event = on_event

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if await self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if self.on_event:
            await self.on_event("plan", plan.rationale)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.tasks = plan.tasks

        max_parallel = self.client.settings.max_parallel_tasks
        if max_parallel and max_parallel > 0:
            self._semaphore = asyncio.Semaphore(max_parallel)

        async def run_task(task):
            if self._semaphore:
                async with self._semaphore:
                    if self.on_event:
                        await self.on_event("task_start", task.id)
                    result = await WorkerAgent(task=task, client=self.client).run()
                    if self.on_event:
                        await self.on_event("task_complete", f"{task.id}: {result}")
                    return result
            else:
                if self.on_event:
                    await self.on_event("task_start", task.id)
                result = await WorkerAgent(task=task, client=self.client).run()
                if self.on_event:
                    await self.on_event("task_complete", f"{task.id}: {result}")
                return result

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                if self.on_event:
                    await self.on_event("task_failed", f"{task.id}: {result}")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            if self.on_event:
                await self.on_event("test", f"passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            if self.on_event:
                await self.on_event("test", f"failed: {test_result.summary}")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        failed = any(t.status == "failed" for t in tasks)
        if failed:
            return False
        all_completed = all(t.status == "completed" for t in tasks)
        if all_completed:
            if self.on_event:
                await self.on_event("done", "all tasks completed")
            return True
        max_cycles = self.client.settings.max_cycles
        if max_cycles and max_cycles > 0 and self.cycles >= max_cycles:
            if self.on_event:
                await self.on_event("done", f"max cycles ({max_cycles}) reached")
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.tasks
