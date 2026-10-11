from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, settings
from furrow.llm import LLMClient

console = Console()

STATE_FILE = Path.cwd() / ".furrow_state.json"


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings_obj: Settings | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.settings = settings_obj or settings
        self.all_tasks: list[Any] = []
        self._load_state()

    def _load_state(self) -> None:
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    data = json.load(f)
                self.cycles = data.get("cycles", 0)
                self.all_tasks = data.get("tasks", [])
                if data.get("goal"):
                    self.goal = data["goal"]
                console.print(f"[dim]Loaded state: cycle {self.cycles}, {len(self.all_tasks)} tasks[/dim]")
            except Exception as e:
                console.print(f"[yellow]Failed to load state: {e}[/yellow]")

    def _save_state(self) -> None:
        try:
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [
                    {
                        "id": t.id,
                        "description": t.description,
                        "files": t.files,
                        "dependencies": t.dependencies,
                        "status": t.status,
                        "result": t.result,
                    }
                    for t in self.all_tasks
                ],
            }
            with open(STATE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            console.print(f"[yellow]Failed to save state: {e}[/yellow]")

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            # Check max cycles
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Stopping.[/yellow]")
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._clear_state()
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, self.all_tasks)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to all_tasks
        for task in plan.tasks:
            self.all_tasks.append(task)

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

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        if failed > 0:
            return False
        return bool(pending == 0 and completed == len(self.all_tasks))

    def _clear_state(self) -> None:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
