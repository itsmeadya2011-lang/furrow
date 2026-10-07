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
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()

STATE_FILE = ".furrow_state.json"


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, max_cycles: int = 10, resume: bool = False) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = max_cycles
        self.all_tasks: list[TaskModel] = []
        self.task_history: list[dict[str, Any]] = []
        self.state_file = Path.cwd() / STATE_FILE
        
        if resume:
            self._load_state()

    def _load_state(self) -> None:
        """Load state from checkpoint file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                self.cycles = state.get("cycles", 0)
                self.goal = state.get("goal", self.goal)
                self.task_history = state.get("task_history", [])
                # Reconstruct tasks from history
                task_map = {}
                for cycle in self.task_history:
                    for task_data in cycle.get("tasks", []):
                        task_id = task_data["task_id"]
                        if task_id not in task_map:
                            task_map[task_id] = TaskModel(
                                id=task_id,
                                description="",  # Will be filled from plan
                                status=task_data["status"],
                                result=task_data.get("result")
                            )
                self.all_tasks = list(task_map.values())
                console.print(f"[green]Resumed from checkpoint: {self.cycles} cycles completed[/green]")
            except Exception as e:
                console.print(f"[yellow]Failed to load state: {e}[/yellow]")

    def _save_state(self) -> None:
        """Save state to checkpoint file."""
        try:
            state = {
                "cycles": self.cycles,
                "goal": self.goal,
                "task_history": self.task_history,
            }
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            console.print(f"[yellow]Failed to save state: {e}[/yellow]")

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        try:
            while True:
                self.cycles += 1
                if self.cycles > self.max_cycles:
                    console.print(f"[bold red]Max cycles ({self.max_cycles}) reached. Stopping.[/bold red]")
                    break
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                await self._cycle()
                self._save_state()
                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    # Clean up state file on success
                    if self.state_file.exists():
                        self.state_file.unlink()
                    break
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. State saved for resume.[/yellow]")
            self._save_state()
            raise

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self.task_history)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to our tracking
        for task in plan.tasks:
            if not any(t.id == task.id for t in self.all_tasks):
                self.all_tasks.append(task)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        cycle_results = []
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                cycle_results.append({"task_id": task.id, "status": "failed", "error": str(result)})
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                cycle_results.append({"task_id": task.id, "status": "completed", "result": result})

        self.task_history.append({
            "cycle": self.cycles,
            "goal": self.goal,
            "tasks": cycle_results,
            "plan_rationale": plan.rationale
        })

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

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
