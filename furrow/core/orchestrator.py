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
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, on_output: Callable[[str], Awaitable[None] | None] | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list = []
        self.last_test_passed: bool = False
        self.last_test_summary: str = ""
        self.on_output = on_output

    async def _emit(self, message: str) -> None:
        console.print(message)
        if self.on_output is not None:
            result = self.on_output(message)
            if inspect.isawaitable(result):
                await result

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        await self._emit(f"Furrow\nGoal: {self.goal}")
        while True:
            self.cycles += 1
            if self.client.settings.max_cycles > 0 and self.cycles > self.client.settings.max_cycles:
                console.print("[yellow]Reached max_cycles limit. Halting.[/yellow]")
                await self._emit("Reached max_cycles limit. Halting.")
                break
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._emit(f"\n═══ Cycle {self.cycles} ═══")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._emit("Goal complete. Halting.")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        plan_text = f"Plan rationale: {plan.rationale}\nTasks:\n" + "\n".join(f"- {t.description}" for t in plan.tasks)
        await self._emit(plan_text)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._emit("No tasks planned. Goal may be complete.")
            return

        self.tasks = plan.tasks

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            sem = asyncio.Semaphore(self.client.settings.max_parallel_tasks)
            async def _run_one(task):
                async with sem:
                    return await WorkerAgent(task=task, client=self.client).run()
            tasks = [_run_one(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._emit(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._emit(f"Task {task.id} completed")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            await self._emit(f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            await self._emit(f"Tests failed: {test_result.summary}")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
                await self._emit(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            await self._emit("Will attempt fix in next cycle.")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        self.last_test_passed = test_result.passed
        self.last_test_summary = test_result.summary

    def _is_done(self) -> bool:
        return bool(self.tasks) and all(t.status == "completed" for t in self.tasks) and self.last_test_passed

    def _get_tasks(self) -> list[Any]:
        return self.tasks
