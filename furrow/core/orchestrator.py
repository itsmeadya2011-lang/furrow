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
from furrow.config import Plan, TestResult, settings
from furrow.llm import LLMClient
from furrow.logging import logger

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, callbacks: list[Callable] | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self._plan: Plan | None = None
        self._callbacks = callbacks or []

    def _emit(self, msg: dict) -> None:
        for cb in self._callbacks:
            cb(msg)

    async def run(self) -> None:
        self._emit({"type": "start", "message": f"Goal: {self.goal}"})
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        try:
            while True:
                self.cycles += 1
                if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                    console.print("[yellow]Reached max cycles limit.[/yellow]")
                    break
                logger.info("cycle_start", cycle=self.cycles)
                self._emit({"type": "cycle_start", "cycle": self.cycles})
                await self._cycle()
                if self._is_done():
                    logger.info("goal_complete")
                    break
        except Exception as e:
            self._emit({"type": "error", "message": str(e)})
            raise
        self._emit({"type": "done"})

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit({"type": "plan", "tasks": [t.model_dump() for t in plan.tasks], "rationale": plan.rationale})

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

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
            self._emit({"type": "task_done", "task_id": task.id, "status": task.status, "result": task.result})

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await self.tester.run(self.goal, plan.tasks)

        self._emit({"type": "test_result", "passed": test_result.passed, "summary": test_result.summary, "failures": test_result.failures})

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._get_tasks()):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self._plan.tasks if self._plan else []
