from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import structlog
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
logger = structlog.get_logger(__name__)

STATE_FILE = ".furrow_state.json"


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._state_path = Path(self.client.settings.workspace) / STATE_FILE
        self._load_state()

    def _load_state(self) -> None:
        if not self._state_path.exists():
            logger.info("No existing state file found, starting fresh")
            return
        try:
            data = json.loads(self._state_path.read_text())
            self.cycles = data.get("cycles", 0)
            self.goal = data.get("goal", self.goal)
            self._tasks = [TaskModel(**t) for t in data.get("tasks", [])]
            logger.info("State loaded", cycles=self.cycles, tasks=len(self._tasks))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to load state, starting fresh", error=str(e))
            self._tasks = []
            self.cycles = 0

    def _save_state(self) -> None:
        data = {
            "cycles": self.cycles,
            "goal": self.goal,
            "tasks": [t.model_dump() for t in self._tasks],
        }
        try:
            self._state_path.write_text(json.dumps(data, indent=2))
            logger.debug("State saved", path=str(self._state_path), cycles=self.cycles)
        except OSError as e:
            logger.error("Failed to save state", error=str(e))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.client.settings.max_cycles

        while True:
            if max_cycles > 0 and self.cycles >= max_cycles:
                logger.warning("Max cycles reached, stopping", max_cycles=max_cycles, current=self.cycles)
                console.print(f"[yellow]Max cycles ({max_cycles}) reached. Stopping.[/yellow]")
                break

            self.cycles += 1
            logger.info("Starting cycle", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            try:
                await self._cycle()
            except Exception as e:
                logger.exception("Cycle failed", cycle=self.cycles, error=str(e))
                console.print(f"[red]Cycle {self.cycles} failed: {e}[/red]")

            self._save_state()

            if self._is_done():
                logger.info("Goal complete, halting", cycle=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)

        logger.info("Plan received", task_count=len(plan.tasks))
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.info("No tasks planned, goal may be complete")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        new_tasks = []
        for task in plan.tasks:
            existing = next((t for t in self._tasks if t.id == task.id), None)
            if existing:
                existing.description = task.description
                existing.files = task.files
                existing.dependencies = task.dependencies
                existing.status = "pending"
                existing.result = None
                new_tasks.append(existing)
            else:
                new_tasks.append(task)

        self._tasks.extend([t for t in new_tasks if t not in self._tasks])

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks_to_run = [t for t in new_tasks if t.status == "pending"]
            if not tasks_to_run:
                logger.info("No pending tasks to execute")
            else:
                tasks = [WorkerAgent(task=task, client=self.client).run() for task in tasks_to_run]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for task, result in zip(tasks_to_run, results):
                    if isinstance(result, Exception):
                        task.status = "failed"
                        task.result = str(result)
                        logger.error("Task failed", task_id=task.id, error=str(result))
                        console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    else:
                        task.status = "completed"
                        task.result = result
                        logger.info("Task completed", task_id=task.id)
                        console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self._tasks)

        if test_result.passed:
            logger.info("Tests passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("Tests failed", failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self._tasks:
            return False

        completed = sum(1 for t in self._tasks if t.status == "completed")
        failed = sum(1 for t in self._tasks if t.status == "failed")
        pending = sum(1 for t in self._tasks if t.status == "pending")

        if failed > 0:
            return False

        if completed >= len(self._tasks) and pending == 0:
            return True

        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks