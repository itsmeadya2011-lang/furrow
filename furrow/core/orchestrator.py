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

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int | None = None,
        session_path: str | Path | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = max_cycles or settings.max_cycles
        self.session_path = Path(session_path) if session_path else Path.cwd() / ".furrow_session.json"
        self.all_tasks: list[TaskModel] = []
        self.cycle_history: list[dict[str, Any]] = []
        self._load_session()

    def _load_session(self) -> None:
        if self.session_path.exists():
            try:
                data = json.loads(self.session_path.read_text())
                self.cycles = data.get("cycles", 0)
                self.all_tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                self.cycle_history = data.get("history", [])
                self.goal = data.get("goal", self.original_goal)
                console.print(f"[dim]Resumed session: {self.cycles} cycles, {len(self.all_tasks)} tasks[/dim]")
            except Exception as e:
                console.print(f"[yellow]Could not load session: {e}[/yellow]")

    def _save_session(self) -> None:
        data = {
            "original_goal": self.original_goal,
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.all_tasks],
            "history": self.cycle_history,
        }
        self.session_path.write_text(json.dumps(data, indent=2))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}", title="Furrow"))
        if self.max_cycles:
            console.print(f"[dim]Max cycles: {self.max_cycles}[/dim]")
        while True:
            if self.max_cycles and self.cycles >= self.max_cycles:
                console.print(f"[yellow]Max cycles ({self.max_cycles}) reached. Halting.[/yellow]")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_session()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        cycle_start_tasks = len(self.all_tasks)
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self.all_tasks, self.cycle_history)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, workspace=settings.workspace).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        cycle_tasks: list[TaskModel] = []
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
            cycle_tasks.append(task)
            self.all_tasks.append(task)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await self.tester.run(self.original_goal, cycle_tasks)

        cycle_record = {
            "cycle": self.cycles,
            "goal": self.goal,
            "plan": plan.model_dump(),
            "tasks": [t.model_dump() for t in cycle_tasks],
            "test_result": test_result.model_dump(),
        }
        self.cycle_history.append(cycle_record)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Original goal: {self.original_goal}\n\nFix failing tests from cycle {self.cycles}:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.all_tasks):
            return True
        return False
