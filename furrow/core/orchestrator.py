from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

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
        on_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._current_tasks: list[Any] = []
        self.on_event = on_event

    async def _emit(self, event: dict[str, Any]) -> None:
        """Emit an event to the on_event callback if one is registered."""
        if self.on_event is not None:
            await self.on_event(event)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

        await self._emit({"type": "done", "message": "Goal complete"})

    async def _cycle(self) -> None:
        await self._emit({"type": "phase", "phase": "planning"})
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit({
            "type": "plan",
            "tasks": [t.model_dump() for t in plan.tasks],
            "rationale": plan.rationale,
        })

        if not plan.tasks:
            self._current_tasks = []
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self._current_tasks = plan.tasks

        await self._emit({"type": "phase", "phase": "executing"})
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
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
            await self._emit({
                "type": "task_update",
                "id": task.id,
                "status": task.status,
                "result": task.result,
            })

        await self._emit({"type": "phase", "phase": "testing"})
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        await self._emit({
            "type": "test",
            "passed": test_result.passed,
            "summary": test_result.summary,
            "failures": test_result.failures,
        })

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        return all(t.status == "completed" for t in tasks)

    def _get_tasks(self) -> list[Any]:
        return self._current_tasks
