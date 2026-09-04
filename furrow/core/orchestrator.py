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
from structlog import get_logger

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

_console = Console()
logger = get_logger(__name__)

EventCallback = Callable[[str, dict], None]


def _noop_event(event_type: str, data: dict) -> None:
    return None


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: EventCallback | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.current_plan: Plan | None = None
        self.on_event: EventCallback = on_event or _noop_event
        self.console: Console = console or _console

    def _emit(self, event_type: str, data: dict) -> None:
        try:
            self.on_event(event_type, data)
        except Exception:
            logger.exception("on_event_callback_failed", event_type=event_type)

    async def run(self) -> None:
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._emit("goal_complete", {"cycles": self.cycles})
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console) as status:
            plan = await self.planner.plan(self.goal)
        self.current_plan = plan
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_generated", cycle=self.cycles, tasks=len(plan.tasks), rationale=plan.rationale)
        self._emit("plan_generated", plan.model_dump())

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.warning("no_tasks_planned", cycle=self.cycles)
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            for task in plan.tasks:
                self._emit("task_started", {"id": task.id, "description": task.description})
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", cycle=self.cycles, task_id=task.id, error=str(result))
                self._emit("task_failed", {"id": task.id, "description": task.description, "error": str(result)})
            else:
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", cycle=self.cycles, task_id=task.id)
                self._emit("task_completed", {"id": task.id, "description": task.description, "result": result})

        with Status("[bold yellow]Testing...", console=self.console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
            self._emit("tests_passed", {"summary": test_result.summary})
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", cycle=self.cycles, summary=test_result.summary, failures=test_result.failures)
            self._emit("tests_failed", {"summary": test_result.summary, "failures": list(test_result.failures)})
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
            logger.info("max_cycles_reached", cycles=self.cycles, max_cycles=self.client.settings.max_cycles)
            return True
        tasks = self._get_tasks()
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            logger.debug("not_done", reason="failed_tasks", failed=failed, completed=completed, total=len(tasks))
            return False
        if completed >= len(tasks):
            logger.info("goal_complete", cycles=self.cycles, completed=completed, total=len(tasks))
            return True
        logger.debug("not_done", reason="incomplete_tasks", completed=completed, total=len(tasks))
        return False

    def _get_tasks(self) -> list[Any]:
        if self.current_plan is None:
            return []
        return self.current_plan.tasks
