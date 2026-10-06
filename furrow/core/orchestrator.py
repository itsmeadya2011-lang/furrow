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
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.llm import LLMClient
from furrow.logging import get_logger, setup_logging

setup_logging()
default_console = Console()
logger = get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        state_file: Path | None = None,
        max_cycles: int | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = max_cycles or settings.max_cycles
        self.state_file = state_file or (settings.workspace / ".furrow_state.json")
        self.all_tasks: list[TaskModel] = []
        self.console = console or default_console
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self.all_tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                self.console.print(f"[dim]Loaded state: {len(self.all_tasks)} tasks, cycle {self.cycles}[/dim]")
            except Exception as e:
                self.console.print(f"[yellow]Could not load state: {e}[/yellow]")

    def _save_state(self) -> None:
        data = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.all_tasks],
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    async def run(self) -> None:
        logger.info("furrow_started", goal=self.goal, max_cycles=self.max_cycles)
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.max_cycles and self.cycles >= self.max_cycles:
                logger.info("max_cycles_reached", max_cycles=self.max_cycles)
                self.console.print(f"[yellow]Max cycles ({self.max_cycles}) reached. Stopping.[/yellow]")
                break
            self.cycles += 1
            logger.info("cycle_started", cycle=self.cycles)
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                logger.info("goal_complete", cycles=self.cycles, total_tasks=len(self.all_tasks))
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        logger.info("planning_started", cycle=self.cycles)
        with Status("[bold yellow]Planning...", console=self.console) as status:
            plan = await self.planner.plan(self.goal)
        logger.info("planning_completed", cycle=self.cycles, task_count=len(plan.tasks))
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.warning("no_tasks_planned", cycle=self.cycles)
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Track new tasks
        for task in plan.tasks:
            task.status = "pending"
            self.all_tasks.append(task)

        logger.info("execution_started", cycle=self.cycles, task_count=len(plan.tasks))
        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        completed_count = 0
        failed_count = 0
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                failed_count += 1
                logger.error("task_failed", task_id=task.id, error=str(result))
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                completed_count += 1
                logger.info("task_completed", task_id=task.id)
                self.console.print(f"[green]Task {task.id} completed[/green]")

        logger.info("execution_completed", cycle=self.cycles, completed=completed_count, failed=failed_count)

        logger.info("testing_started", cycle=self.cycles)
        with Status("[bold yellow]Testing...", console=self.console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("tests_failed", cycle=self.cycles, failures=test_result.failures)
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        if failed > 0:
            return False
        if completed > 0 and pending == 0:
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
