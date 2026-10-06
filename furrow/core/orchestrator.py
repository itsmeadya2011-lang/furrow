from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.original_goal = goal
        self.current_goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.cycle_history: list[dict[str, Any]] = []
        self.workspace = self.settings.workspace

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}", title="Furrow"))
        
        while True:
            self.cycles += 1
            
            # Check max_cycles limit
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Stopping.[/yellow]")
                break
            
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.current_goal)
        
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Execute tasks respecting dependencies
        cycle_tasks = await self._execute_tasks_with_dependencies(plan.tasks)
        self.all_tasks.extend(cycle_tasks)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await self.tester.run(self.current_goal, self.all_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Append failures to goal for next cycle
            self.current_goal = f"{self.original_goal}\n\nPrevious failures to fix:\n" + "\n".join(test_result.failures)

        # Record cycle history
        self.cycle_history.append({
            "cycle": self.cycles,
            "goal": self.current_goal,
            "tasks": [t.model_dump() for t in cycle_tasks],
            "test_result": test_result.model_dump(),
        })

        # Git checkpoint after successful cycle
        if test_result.passed and self.settings.enable_git_checkpoint:
            await self._git_checkpoint()

    async def _git_checkpoint(self) -> None:
        """Create a git commit checkpoint after successful cycle."""
        try:
            # Check if we're in a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.workspace,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                return  # Not a git repo
            
            # Check if there are changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.workspace,
                capture_output=True,
                text=True
            )
            if not result.stdout.strip():
                return  # No changes
            
            # Stage all changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.workspace,
                capture_output=True
            )
            
            # Create commit
            commit_msg = f"furrow: cycle {self.cycles} - {self.original_goal[:50]}"
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=self.workspace,
                capture_output=True
            )
            console.print(f"[dim]Git checkpoint created: {commit_msg}[/dim]")
        except Exception as e:
            console.print(f"[dim]Git checkpoint failed: {e}[/dim]")

    async def _execute_tasks_with_dependencies(self, tasks: list[TaskModel]) -> list[TaskModel]:
        """Execute tasks respecting their dependencies."""
        completed_ids = set()
        results = {}
        
        # Keep running until all tasks are done or no progress can be made
        while len(completed_ids) < len(tasks):
            # Find tasks that can run (dependencies met)
            runnable = [
                task for task in tasks
                if task.id not in completed_ids
                and all(dep in completed_ids for dep in task.dependencies)
            ]
            
            if not runnable:
                # Circular dependency or missing dependency - run remaining anyway
                runnable = [task for task in tasks if task.id not in completed_ids]
                console.print("[yellow]Warning: Possible circular dependency, running remaining tasks[/yellow]")
            
            # Limit parallel execution
            runnable = runnable[:self.settings.max_parallel_tasks]
            
            with Status(f"[bold yellow]Executing {len(runnable)} tasks...", console=console):
                worker_tasks = [
                    WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                    for task in runnable
                ]
                task_results = await asyncio.gather(*worker_tasks, return_exceptions=True)
            
            for task, result in zip(runnable, task_results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                completed_ids.add(task.id)
                results[task.id] = task
        
        return [results[task.id] for task in tasks]

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

    def get_status(self) -> dict[str, Any]:
        """Get current orchestrator status for web API."""
        return {
            "cycles": self.cycles,
            "original_goal": self.original_goal,
            "current_goal": self.current_goal,
            "tasks": [t.model_dump() for t in self.all_tasks],
            "completed": sum(1 for t in self.all_tasks if t.status == "completed"),
            "failed": sum(1 for t in self.all_tasks if t.status == "failed"),
            "pending": sum(1 for t in self.all_tasks if t.status == "pending"),
        }