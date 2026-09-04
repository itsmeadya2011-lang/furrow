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
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
logger = get_logger("orchestrator")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._current_tasks: list[Any] = []
        self._on_event = on_event

    def _emit(self, type: str, message: str = "", data: Any = None) -> None:
        if self._on_event is None:
            return
        payload: dict[str, Any] = {"type": type, "message": message}
        if data is not None:
            payload["data"] = data
        self._on_event(payload)

    async def run(self) -> None:
        logger.info("orchestrator_started", goal=self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        self._emit("status", "Orchestrator started")
        while True:
            self.cycles += 1
            logger.info("cycle_started", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit("status", f"Cycle {self.cycles} started")
            await self._cycle()
            if self._is_done():
                logger.info("goal_complete", goal=self.original_goal)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._emit("goal_complete", "Goal complete")
                break
            logger.info("cycle_ended", cycle=self.cycles)

    async def _cycle(self) -> None:
        logger.debug("planning_started", cycle=self.cycles)
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        logger.debug("planning_complete", cycle=self.cycles, tasks=len(plan.tasks))
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit("plan", "Planning complete", data=plan.model_dump())

        if not plan.tasks:
            logger.debug("no_tasks_planned", cycle=self.cycles)
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._emit("status", "No tasks planned")
            return

        self._current_tasks = plan.tasks

        logger.debug("task_execution_started", cycle=self.cycles, tasks=len(plan.tasks))
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
                logger.error("task_failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                logger.info("task_completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")
                self._emit("task_completed", f"Task {task.id} completed")
        logger.debug("task_execution_complete", cycle=self.cycles)

        logger.debug("testing_started", cycle=self.cycles)
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            logger.info("testing_complete", passed=True, summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._emit("testing_complete", f"Tests passed: {test_result.summary}")
        else:
            logger.info("testing_complete", passed=False, summary=test_result.summary)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self._emit("testing_complete", f"Tests failed: {test_result.summary}")
            self.goal = (
                f"{self.original_goal}\n\n"
                f"Previous cycle completed. Fix these test failures:\n"
                + "\n".join(test_result.failures)
            )

    def _is_done(self) -> bool:
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._get_tasks()):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self._current_tasks
