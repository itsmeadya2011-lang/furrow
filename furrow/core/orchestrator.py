from __future__ import annotations

import asyncio
import json
import os
import uuid
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

STATE_FILE = Path(".furrow_state.json")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        config: Settings | None = None,
        state_file: Path | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(config or settings)
        self.config = config or settings
        self.planner = PlannerAgent(client=self.client, settings=self.config)
        self.cycles = 0
        self.state_file = state_file or STATE_FILE
        self.tasks: list[Any] = []
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.cycles = data.get("cycles", 0)
                self.goal = data.get("goal", self.goal)
                self.tasks = data.get("tasks", [])
                console.print(f"[dim]Loaded state: cycle {self.cycles}, {len(self.tasks)} tasks[/dim]")
            except Exception as e:
                console.print(f"[yellow]Failed to load state: {e}[/yellow]")

    def _save_state(self) -> None:
        data = {
            "cycles": self.cycles,
            "goal": self.goal,
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "files": t.files,
                    "dependencies": t.dependencies,
                    "status": t.status,
                    "result": t.result,
                }
                for t in self.tasks
            ],
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        
        # Check if there are pending tasks from previous state
        pending_tasks = [t for t in self.tasks if t.status == "pending"]
        if pending_tasks:
            console.print(f"[cyan]Resuming with {len(pending_tasks)} pending tasks[/cyan]")
        
        while True:
            # Check max cycles
            if self.config.max_cycles > 0 and self.cycles >= self.config.max_cycles:
                console.print(f"[yellow]Max cycles ({self.config.max_cycles}) reached. Stopping.[/yellow]")
                break
            
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._cleanup_state()
                break

    async def _cycle(self) -> None:
        # Get pending tasks from previous cycles
        pending_tasks = [t for t in self.tasks if t.status == "pending"]
        
        if not pending_tasks:
            with Status("[bold yellow]Planning...", console=console) as status:
                plan = await self.planner.plan(self.goal, self._get_project_context())
            console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

            if not plan.tasks:
                console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
                return

            # Add new tasks to our task list
            for task in plan.tasks:
                task.status = "pending"
            self.tasks.extend(plan.tasks)
            pending_tasks = plan.tasks

        # Execute pending tasks in parallel (respecting max_parallel_tasks)
        semaphore = asyncio.Semaphore(self.config.max_parallel_tasks)
        
        async def run_with_semaphore(task):
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client, settings=self.config).run()

        with Status(f"[bold yellow]Executing {len(pending_tasks)} tasks (max parallel: {self.config.max_parallel_tasks})...", console=console):
            tasks_coros = [run_with_semaphore(task) for task in pending_tasks]
            results = await asyncio.gather(*tasks_coros, return_exceptions=True)

        for task, result in zip(pending_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        # Run tests
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.config).run(self.goal, self.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Add fix tasks for failures
            for i, failure in enumerate(test_result.failures):
                fix_task = type(self.tasks[0])(
                    id=f"fix-{self.cycles}-{i}",
                    description=f"Fix test failure: {failure}",
                    files=[],
                    dependencies=[],
                    status="pending",
                    result=None,
                )
                self.tasks.append(fix_task)

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        pending = sum(1 for t in self.tasks if t.status == "pending")
        
        if failed > 0 and pending == 0:
            return False
        if pending > 0:
            return False
        if completed >= len(self.tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.tasks

    def _get_project_context(self) -> str:
        """Get project context for the planner."""
        files = self.client.list_files(self.config.workspace)
        return f"Project files:\n" + "\n".join(files[:100])  # Limit to 100 files

    def _cleanup_state(self) -> None:
        if self.state_file.exists():
            self.state_file.unlink()

    def add_user_input(self, user_input: str) -> None:
        """Incorporate mid-loop user input into the goal."""
        self.goal = f"{self.goal}\n\nUser input: {user_input}"
        # Reset pending tasks to re-plan with new context
        for task in self.tasks:
            if task.status == "pending":
                task.status = "cancelled"
