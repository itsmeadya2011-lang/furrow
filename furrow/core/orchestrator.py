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
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.core.state import StateManager
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        orchestrator_settings: Settings | None = None,
        resume: bool = False,
    ) -> None:
        self.settings = orchestrator_settings or settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.state_manager = StateManager(workspace=self.settings.workspace, settings=self.settings)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.cycle_history: list[dict[str, Any]] = []

        # Load state if resuming
        if resume:
            loaded = self.state_manager.load()
            if loaded:
                self.goal = loaded.get("goal", goal)
                self.cycles = loaded.get("cycles", 0)
                self.all_tasks = [TaskModel(**t) for t in loaded.get("all_tasks", [])]
                self.cycle_history = loaded.get("cycle_history", [])
                console.print(f"[green]Resumed from cycle {self.cycles} with {len(self.all_tasks)} tasks[/green]")
            else:
                self.goal = goal
                console.print("[yellow]No saved state found, starting fresh[/yellow]")
        else:
            self.goal = goal

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))

        max_cycles = self.settings.max_cycles
        if max_cycles > 0:
            console.print(f"[dim]Max cycles: {max_cycles}[/dim]")

        try:
            while True:
                if max_cycles > 0 and self.cycles >= max_cycles:
                    console.print(f"[yellow]Max cycles ({max_cycles}) reached. Halting.[/yellow]")
                    break

                self.cycles += 1
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

                cycle_result = await self._cycle()
                self.cycle_history.append(cycle_result)

                # Save state after each cycle
                self.state_manager.save(self.goal, self.all_tasks, self.cycle_history, self.cycles)

                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    self.state_manager.clear()
                    break

        except KeyboardInterrupt:
            console.print("\n[yellow]Furrow stopped by user. State saved.[/yellow]")
            self.state_manager.save(self.goal, self.all_tasks, self.cycle_history, self.cycles)
            raise

        self._print_summary()

    async def _cycle(self) -> dict[str, Any]:
        # Plan
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self.all_tasks, self.cycle_history)

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return {"cycle": self.cycles, "plan": plan.model_dump(), "tasks": [], "test_result": None}

        # Filter out already completed tasks
        new_tasks = [t for t in plan.tasks if t.id not in [ct.id for ct in self.all_tasks if ct.status == "completed"]]
        if not new_tasks:
            console.print("[yellow]All planned tasks already completed.[/yellow]")
            return {"cycle": self.cycles, "plan": plan.model_dump(), "tasks": [], "test_result": None}

        # Execute tasks in parallel with semaphore for max parallelism
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def run_with_semaphore(task: TaskModel) -> TaskModel:
            async with semaphore:
                worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
                try:
                    result = await worker.run()
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                except Exception as e:
                    task.status = "failed"
                    task.result = str(e)
                    console.print(f"[red]Task {task.id} failed: {e}[/red]")
                return task

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [run_with_semaphore(task) for task in new_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update all_tasks with results
        for task, result in zip(new_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
            elif isinstance(result, TaskModel):
                # result is the updated task
                pass
            else:
                task.status = "completed"
                task.result = str(result)

            # Add to all_tasks if not already present
            if not any(t.id == task.id for t in self.all_tasks):
                self.all_tasks.append(task)

        # Test
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

        return {
            "cycle": self.cycles,
            "plan": plan.model_dump(),
            "tasks": [t.model_dump() for t in new_tasks],
            "test_result": test_result.model_dump(),
        }

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False

        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")

        if failed > 0 and pending == 0:
            # All tasks done but some failed - not complete
            return False

        if completed >= len(self.all_tasks) and pending == 0:
            return True

        return False

    def _print_summary(self) -> None:
        total = len(self.all_tasks)
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")

        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  Cycles: {self.cycles}")
        console.print(f"  Total tasks: {total}")
        console.print(f"  Completed: {completed}")
        console.print(f"  Failed: {failed}")
        console.print(f"  Pending: {pending}")