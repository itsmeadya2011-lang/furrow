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
from furrow.config import Plan, TestResult, TaskModel
from furrow.config import settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        emit: callable[[str], None] | None = None,
        state_path: Path | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.emit = emit
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.state_path = state_path or settings.workspace / ".furrow_state.json"
        self._load_state()

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        data = json.loads(self.state_path.read_text())
        self.goal = data["goal"]
        self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]
        self.cycles = data.get("cycles", 0)
        self._log(f"Loaded state from {self.state_path}")

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "goal": self.goal,
            "tasks": [t.model_dump() for t in self.tasks],
            "cycles": self.cycles,
        }
        self.state_path.write_text(json.dumps(data, indent=2))
        self._log(f"Saved state to {self.state_path}")

    def _log(self, message: str) -> None:
        if self.emit is not None:
            self.emit(message)
        else:
            console.print(message)

    async def run(self) -> None:
        self._log(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._log(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                self._log("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                break
            if self._is_done():
                self._log("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._log(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._log("[yellow]No tasks planned. Goal may be complete.[/yellow]")
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
                self._log(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._log(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self._log(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._log(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._log(f"  • {failure}")
            self._log("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        self.tasks = list(plan.tasks)
        self._save_state()

    def _is_done(self) -> bool:
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._get_tasks()):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.tasks