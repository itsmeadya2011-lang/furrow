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
from furrow.config import Plan, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self.workspace = settings.workspace
        self.state_path = self.workspace / ".furrow" / "state.json"
        self._load_state()

    def _load_state(self) -> None:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                self.cycles = data.get("cycle", 0)
                if data.get("goal"):
                    self.goal = data["goal"]
                console.print(f"[dim]Loaded state from {self.state_path} (cycle {self.cycles})[/dim]")
            except (json.JSONDecodeError, OSError):
                pass

    def _save_state(self) -> None:
        state = {
            "cycle": self.cycles,
            "goal": self.goal,
            "tasks": [
                {
                    "id": t.id,
                    "status": t.status,
                    "result": t.result,
                }
                for t in self._get_tasks()
            ],
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if 0 < settings.max_cycles <= self.cycles:
                console.print(f"[yellow]Reached max_cycles ({settings.max_cycles}). Halting.[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            self.plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(self.plan.model_dump()), title="Plan", border_style="blue"))

        if not self.plan.tasks:
            if self.cycles > 0 and not self._get_tasks():
                self.goal = f"Planning returned no tasks for goal:\n{self.goal}\nPlease break the goal into actionable implementation tasks."
                console.print("[yellow]Planner returned no tasks. Refining goal for next cycle.[/yellow]")
            else:
                console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._save_state()
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in self.plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(self.plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        console.print(f"[bold]Cycle {self.cycles} results:[/bold] {completed} completed, {failed} failed, {len(self._get_tasks()) - completed - failed} pending")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        self._save_state()

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        return completed == len(tasks) and failed == 0

    def _get_tasks(self) -> list[Any]:
        if self.plan is None:
            return []
        return self.plan.tasks
