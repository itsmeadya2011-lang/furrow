from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        config: Settings | None = None,
        resume: bool = False,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient(config or settings)
        self.config = config or settings
        self.planner = PlannerAgent(client=self.client, settings=self.config)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.cycle_history: list[dict[str, Any]] = []
        self.resume = resume
        self.state_file = Path(self.config.workspace) / ".furrow_state.json"
        
        if resume and self.state_file.exists():
            self._load_state()

    async def run(self) -> None:
        console.print(Panel.fit(
            f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}\nCycles: {self.cycles}/{self.config.max_cycles or '∞'}",
            title="Furrow"
        ))
        
        while True:
            if self.config.max_cycles and self.cycles >= self.config.max_cycles:
                console.print(f"[yellow]Max cycles ({self.config.max_cycles}) reached. Halting.[/yellow]")
                break
                
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        cycle_start = datetime.now()
        
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.cycle_history.append({
                "cycle": self.cycles,
                "goal": self.goal,
                "tasks": [],
                "duration": (datetime.now() - cycle_start).total_seconds(),
                "test_passed": None,
            })
            return

        # Limit parallel tasks
        semaphore = asyncio.Semaphore(self.config.max_parallel_tasks)
        
        async def run_task(task: TaskModel):
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client, settings=self.config).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [run_task(task) for task in plan.tasks]
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
            # Add to all_tasks, updating if already exists
            existing = next((t for t in self.all_tasks if t.id == task.id), None)
            if existing:
                existing.status = task.status
                existing.result = task.result
            else:
                self.all_tasks.append(task)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.config).run(self.goal, plan.tasks)

        cycle_duration = (datetime.now() - cycle_start).total_seconds()
        
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Preserve original goal but add failure context
            self.goal = f"{self.original_goal}\n\nPrevious cycle test failures to fix:\n" + "\n".join(test_result.failures)

        self.cycle_history.append({
            "cycle": self.cycles,
            "goal": self.goal,
            "tasks": [t.model_dump() for t in plan.tasks],
            "duration": cycle_duration,
            "test_passed": test_result.passed,
            "test_summary": test_result.summary,
            "test_failures": test_result.failures,
        })

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
            
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        
        # Show progress
        self._print_progress(completed, failed, pending)
        
        if failed > 0:
            return False
        if completed >= len(self.all_tasks) and pending == 0:
            return True
        return False

    def _print_progress(self, completed: int, failed: int, pending: int) -> None:
        table = Table(title="Task Progress", show_header=True)
        table.add_column("Status", style="cyan")
        table.add_column("Count", style="green")
        table.add_row("Completed", str(completed))
        table.add_row("Failed", str(failed))
        table.add_row("Pending", str(pending))
        table.add_row("Total", str(len(self.all_tasks)))
        console.print(table)

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks

    def _save_state(self) -> None:
        state = {
            "original_goal": self.original_goal,
            "current_goal": self.goal,
            "cycles": self.cycles,
            "all_tasks": [t.model_dump() for t in self.all_tasks],
            "cycle_history": self.cycle_history,
        }
        self.state_file.write_text(json.dumps(state, indent=2))

    def _load_state(self) -> None:
        try:
            data = json.loads(self.state_file.read_text())
            self.original_goal = data.get("original_goal", self.original_goal)
            self.goal = data.get("current_goal", self.goal)
            self.cycles = data.get("cycles", 0)
            self.all_tasks = [TaskModel(**t) for t in data.get("all_tasks", [])]
            self.cycle_history = data.get("cycle_history", [])
            console.print(f"[green]Resumed from cycle {self.cycles}[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not resume state: {e}[/yellow]")
