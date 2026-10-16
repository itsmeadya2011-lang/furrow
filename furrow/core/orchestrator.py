from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable

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
        send: Callable[[str], Any] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.send = send

    async def _emit(self, message: str) -> None:
        if self.send is not None:
            await self.send(message)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._emit(f"Cycle {self.cycles} started")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._emit("Goal complete. Halting.")
                break
            if (
                self.client.settings.max_cycles > 0
                and self.cycles >= self.client.settings.max_cycles
            ):
                console.print("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                await self._emit("Max cycles reached. Halting.")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit(f"Plan: {plan.rationale} ({len(plan.tasks)} tasks)")

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._emit("No tasks planned.")
            return

        self._merge_tasks(plan.tasks)

        max_parallel = self.client.settings.max_parallel_tasks
        pending = [t for t in self.tasks if t.status == "pending"]

        for i in range(0, len(pending), max_parallel):
            chunk = pending[i : i + max_parallel]
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                workers = [WorkerAgent(task=task, client=self.client).run() for task in chunk]
                results = await asyncio.gather(*workers, return_exceptions=True)

            for task, result in zip(chunk, results):
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
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            await self._emit(f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            await self._emit(f"Tests failed: {test_result.summary}")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _merge_tasks(self, new_tasks: list[TaskModel]) -> None:
        existing = {t.id: t for t in self.tasks}
        merged: list[TaskModel] = []
        seen: set[str] = set()
        for task in new_tasks:
            prev = existing.get(task.id)
            if prev is not None and prev.status == "completed":
                merged.append(prev)
            else:
                merged.append(task)
            seen.add(task.id)
        for t in self.tasks:
            if t.id not in seen:
                merged.append(t)
        self.tasks = merged

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        if any(t.status == "failed" for t in self.tasks):
            return False
        return all(t.status == "completed" for t in self.tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
