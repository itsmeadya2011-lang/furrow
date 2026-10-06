from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient

logger = structlog.get_logger(__name__)
console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings_override: Settings | None = None,
        state_file: str | Path | None = None,
    ) -> None:
        self.goal = goal
        effective_settings = settings_override or settings
        self.client = client or LLMClient(settings=effective_settings)
        self.planner = PlannerAgent(client=self.client, settings=effective_settings)
        self.settings = effective_settings
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.state_file = Path(state_file) if state_file else Path.cwd() / ".furrow_state.json"
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                self.cycles = data.get("cycles", 0)
                self.all_tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                if "goal" in data:
                    self.goal = data["goal"]
                logger.info("Loaded state", cycles=self.cycles, tasks=len(self.all_tasks))
            except Exception as e:
                logger.warning("Failed to load state", error=str(e))

    def _save_state(self) -> None:
        try:
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self.all_tasks],
                "updated_at": time.time(),
            }
            self.state_file.write_text(json.dumps(data, indent=2))
            logger.debug("Saved state", file=str(self.state_file))
        except Exception as e:
            logger.error("Failed to save state", error=str(e))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.settings.max_cycles

        while True:
            if max_cycles and self.cycles >= max_cycles:
                console.print(f"[yellow]Reached max cycles ({max_cycles}). Halting.[/yellow]")
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self.all_tasks)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Filter out already completed tasks
        new_tasks = [t for t in plan.tasks if not self._is_task_completed(t.id)]
        if not new_tasks:
            console.print("[yellow]All planned tasks already completed.[/yellow]")
            return

        # Merge new tasks into all_tasks
        for task in new_tasks:
            if not self._is_task_completed(task.id):
                self.all_tasks.append(task)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks_to_run = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in new_tasks
            ]
            results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

        for task, result in zip(new_tasks, results):
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
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(
                self.goal, self.all_tasks
            )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_task_completed(self, task_id: str) -> bool:
        for t in self.all_tasks:
            if t.id == task_id and t.status == "completed":
                return True
        return False

    def _get_pending_tasks(self) -> list[TaskModel]:
        return [t for t in self.all_tasks if t.status != "completed"]

    def _is_done(self) -> bool:
        pending = self._get_pending_tasks()
        if not pending:
            return True
        # Check if all pending tasks are failed (dead end)
        if all(t.status == "failed" for t in pending):
            return False
        return False