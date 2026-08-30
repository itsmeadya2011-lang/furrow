from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Awaitable

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
        on_event: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._latest_plan: Plan | None = None
        self._on_event = on_event

    async def _emit(self, message: str) -> None:
        if self._on_event:
            await self._on_event(message)

    async def run(self) -> None:
        await self._emit(f"Goal: {self.goal}")
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            max_cycles = self.client.settings.max_cycles
            if max_cycles > 0 and self.cycles > max_cycles:
                msg = f"Reached max_cycles ({max_cycles}). Halting."
                await self._emit(msg)
                console.print(f"[yellow]{msg}[/yellow]")
                break
            cycle_msg = f"═══ Cycle {self.cycles} ═══"
            await self._emit(cycle_msg)
            console.print(f"\n[bold cyan]{cycle_msg}[/bold cyan]")
            await self._cycle()
            if self._is_done():
                await self._emit("Goal complete. Halting.")
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        await self._emit("Planning...")
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._latest_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit(f"Plan: {plan.rationale}")

        if not plan.tasks:
            msg = "No tasks planned. Goal may be complete."
            await self._emit(msg)
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        await self._emit("Executing tasks in parallel...")
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                msg = f"Task {task.id} failed: {result}"
                await self._emit(msg)
                console.print(f"[red]{msg}[/red]")
            else:
                task.status = "completed"
                task.result = result
                msg = f"Task {task.id} completed"
                await self._emit(msg)
                console.print(f"[green]{msg}[/green]")

        await self._emit("Testing...")
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            msg = f"Tests passed: {test_result.summary}"
            await self._emit(msg)
            console.print(f"[green]{msg}[/green]")
        else:
            msg = f"Tests failed: {test_result.summary}"
            await self._emit(msg)
            console.print(f"[red]{msg}[/red]")
            for failure in test_result.failures:
                fail_msg = f"  • {failure}"
                await self._emit(fail_msg)
                console.print(fail_msg)
            fix_msg = "Will attempt fix in next cycle."
            await self._emit(fix_msg)
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        if self._latest_plan is None:
            return []
        return self._latest_plan.tasks
