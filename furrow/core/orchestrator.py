from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Awaitable

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

STATE_FILE = Path(".furrow_state.json")


class Orchestrator:
    def __init__(
        self, 
        goal: str, 
        client: LLMClient | None = None, 
        resume: bool = False,
        on_output: Callable[[str], Awaitable[None]] | None = None
    ) -> None:
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[Any] = []
        self.on_output = on_output
        
        if resume and STATE_FILE.exists():
            self._load_state(goal)
        else:
            self.goal = goal
            self._save_state()

    async def _emit(self, message: str) -> None:
        """Emit message to both console and callback."""
        console.print(message)
        if self.on_output:
            await self.on_output(message)

    def _save_state(self) -> None:
        state = {
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
        STATE_FILE.write_text(json.dumps(state, indent=2))

    def _load_state(self, goal: str) -> None:
        try:
            data = json.loads(STATE_FILE.read_text())
            self.goal = data.get("goal", goal)
            self.cycles = data.get("cycles", 0)
            self.all_tasks = []
            for t in data.get("tasks", []):
                from furrow.config import TaskModel
                task = TaskModel(**t)
                self.all_tasks.append(task)
            await self._emit(f"[green]Resumed from cycle {self.cycles} with {len(self.all_tasks)} tasks[/green]")
        except Exception as e:
            await self._emit(f"[yellow]Could not load state: {e}. Starting fresh.[/yellow]")
            self.goal = goal
            self.cycles = 0
            self.all_tasks = []

    async def run(self) -> None:
        await self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = settings.max_cycles or 0
        
        while True:
            if max_cycles and self.cycles >= max_cycles:
                await self._emit(f"[yellow]Max cycles ({max_cycles}) reached. Halting.[/yellow]")
                break
                
            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            
            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                if STATE_FILE.exists():
                    STATE_FILE.unlink()
                break

    async def _cycle(self) -> None:
        await self._emit("[bold yellow]Planning...[/bold yellow]")
        plan = await self.planner.plan(self.goal)
        await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to our tracked list
        for task in plan.tasks:
            self.all_tasks.append(task)

        await self._emit("[bold yellow]Executing tasks in parallel...[/bold yellow]")
        tasks = [
            WorkerAgent(task=task, client=self.client).run()
            for task in plan.tasks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed: {result}[/green]")

        await self._emit("[bold yellow]Testing...[/bold yellow]")
        test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return True
            
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        
        if failed > 0:
            return False
        if pending == 0 and completed > 0:
            return True
        return False
