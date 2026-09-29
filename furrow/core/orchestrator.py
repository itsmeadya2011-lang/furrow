from __future__ import annotations

import asyncio
import json
import os
import structlog
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

logger = structlog.get_logger(__name__)

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self.on_progress = on_progress
        self._settings = self.client.settings
        self._state_file = self._settings.state_file
        self._load_state()

    def _emit(self, msg: str) -> None:
        logger.info(msg)
        if self.on_progress:
            try:
                self.on_progress(msg)
            except Exception:
                pass
        console.print(msg)

    async def run(self) -> None:
        self._emit(f"Goal: {self.goal}")
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._emit(f"Cycle {self.cycles}")
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                self._emit("Goal complete. Halting.")
                self._clear_state()
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self._settings.max_cycles > 0 and self.cycles >= self._settings.max_cycles:
                self._emit("Max cycles reached. Stopping.")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self._tasks = plan.tasks
        self._save_state()

        if not plan.tasks:
            self._emit("No tasks planned. Goal may be complete.")
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
                self._emit(f"Task {task.id} failed: {result}")
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._emit(f"Task {task.id} completed")
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self._emit(f"Tests passed: {test_result.summary}")
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit(f"Tests failed: {test_result.summary}")
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
                console.print(f"  • {failure}")
            self._emit("Will attempt fix in next cycle.")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._get_tasks()):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self._tasks

    def _save_state(self) -> None:
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self._tasks],
        }
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.warning("Failed to save state", error=str(e))

    def _load_state(self) -> None:
        if self._state_file.exists():
            try:
                data = json.loads(self._state_file.read_text())
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self._tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                logger.info("state_loaded", goal=self.goal, cycles=self.cycles, tasks=len(self._tasks))
            except Exception as e:
                logger.warning("Failed to load state", error=str(e))

    def _clear_state(self) -> None:
        try:
            if self._state_file.exists():
                self._state_file.unlink()
                logger.info("state_cleared")
        except Exception as e:
            logger.warning("Failed to clear state", error=str(e))
