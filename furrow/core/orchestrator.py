from __future__ import annotations

import asyncio
import json
import os
import time
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
from furrow.config import (
    CycleRecord,
    OrchestratorState,
    Plan,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
    settings,
)
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings_obj: Settings = settings,
        resume: bool = False,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings_obj)
        self.settings = settings_obj
        self.planner = PlannerAgent(client=self.client)
        self.state = OrchestratorState(goal=goal)
        self.state_file = settings_obj.state_file
        self.cycles = 0

        if resume:
            self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                self.state = OrchestratorState(**data)
                console.print(f"[green]Resumed from state file: {self.state.cycles_completed} cycles completed[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not load state: {e}[/yellow]")

    def _save_state(self) -> None:
        self.state.updated_at = self.state.updated_at.__class__.now()
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state.model_dump(mode="json"), f, indent=2, default=str)
        except Exception as e:
            console.print(f"[red]Failed to save state: {e}[/red]")

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        
        # Check if we've already hit max_cycles
        if self.settings.max_cycles > 0 and self.state.cycles_completed >= self.settings.max_cycles:
            console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Use --resume to continue.[/yellow]")
            return

        while True:
            # Check cycle limit
            if self.settings.max_cycles > 0 and self.state.cycles_completed >= self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Stopping.[/yellow]")
                break

            self.cycles += 1
            self.state.current_cycle = self.cycles
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            
            cycle_start = time.time()
            await self._cycle()
            cycle_duration = time.time() - cycle_start

            self.state.cycles_completed += 1
            self.state.current_cycle = 0
            
            # Update cycle record with duration
            if self.state.cycles:
                self.state.cycles[-1].duration_seconds = cycle_duration
            
            self._save_state()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        # Get incomplete tasks from previous cycles
        incomplete_tasks = [t for t in self.state.all_tasks if t.status != TaskStatus.COMPLETED]
        
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, incomplete_tasks)
        
        plan.cycle = self.cycles
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._record_cycle(plan, TestResult(passed=True, summary="No tasks planned"))
            return

        # Add new tasks to state
        for task in plan.tasks:
            task.cycle = self.cycles
            task.started_at = task.started_at.__class__.now()
            self.state.all_tasks.append(task)

        self._save_state()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = TaskStatus.FAILED
                task.result = str(result)
                task.completed_at = task.completed_at.__class__.now()
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = task.completed_at.__class__.now()
                console.print(f"[green]Task {task.id} completed[/green]")

        self._save_state()

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        self._record_cycle(plan, test_result)
        self._save_state()
        self._print_progress()

    def _record_cycle(self, plan: Plan, test_result: TestResult) -> None:
        record = CycleRecord(
            cycle=self.cycles,
            plan=plan,
            test_result=test_result,
        )
        self.state.cycles.append(record)

    def _print_progress(self) -> None:
        completed = sum(1 for t in self.state.all_tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.state.all_tasks if t.status == TaskStatus.FAILED)
        pending = sum(1 for t in self.state.all_tasks if t.status == TaskStatus.PENDING)
        in_progress = sum(1 for t in self.state.all_tasks if t.status == TaskStatus.IN_PROGRESS)

        table = Table(title="Progress")
        table.add_column("Status", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_row("Completed", str(completed), style="green")
        table.add_row("Failed", str(failed), style="red")
        table.add_row("In Progress", str(in_progress), style="yellow")
        table.add_row("Pending", str(pending), style="blue")
        console.print(table)

    def _is_done(self) -> bool:
        # Check if all tasks are completed and tests pass
        incomplete = [t for t in self.state.all_tasks if t.status != TaskStatus.COMPLETED]
        if not incomplete:
            # All tasks completed, check last test result
            if self.state.cycles and self.state.cycles[-1].test_result.passed:
                return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.state.all_tasks
