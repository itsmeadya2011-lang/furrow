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
from furrow.config import Plan, TestResult, Settings, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.planner = PlannerAgent(client=self.client, settings=settings)
        self.settings = settings or Settings()
        self.cycles = 0
        self.all_tasks: list[Any] = []
        self.state_file = self.settings.workspace / ".furrow_state.json"
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.all_tasks = data.get("tasks", [])
                    self.cycles = data.get("cycles", 0)
                    console.print(f"[dim]Loaded state: {len(self.all_tasks)} tasks, {self.cycles} cycles[/dim]")
            except Exception:
                pass

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
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to save state: {e}[/yellow]")

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            
            # Check max cycles
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                console.print(f"[yellow]Reached max cycles ({self.settings.max_cycles}). Stopping.[/yellow]")
                break
            
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
        new_tasks = [t for t in plan.tasks if not any(
            existing.id == t.id and existing.status == "completed" for existing in self.all_tasks
        )]
        
        if not new_tasks:
            console.print("[yellow]All planned tasks already completed.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            # Limit parallel tasks
            semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
            
            async def run_with_semaphore(task):
                async with semaphore:
                    return await WorkerAgent(task=task, client=self.client, settings=self.settings).run()
            
            tasks = [run_with_semaphore(task) for task in new_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(new_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
            
            # Add to all_tasks if not already there
            if not any(existing.id == task.id for existing in self.all_tasks):
                self.all_tasks.append(task)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, self.all_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Append failures to goal context rather than replacing
            self.goal = f"{self.goal}\n\nFix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        
        if failed > 0:
            return False
        if pending > 0:
            return False
        if completed >= len(self.all_tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks
