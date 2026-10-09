from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, settings
from furrow.llm import LLMClient

console = Console()

STATE_FILE = ".furrow_state.json"


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._load_state()

    def _load_state(self) -> None:
        """Load state from disk if exists."""
        path = Path(STATE_FILE)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                console.print(f"[dim]Loaded state: {self.cycles} cycles, {len(self.tasks)} tasks[/dim]")
            except (json.JSONDecodeError, OSError, ValueError) as e:
                console.print(f"[yellow]Could not load state: {e}[/yellow]")

    def _save_state(self) -> None:
        """Save state to disk."""
        path = Path(STATE_FILE)
        data = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
        }
        path.write_text(json.dumps(data, indent=2))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._cleanup_state()
                break
            
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                console.print(f"[yellow]Max cycles ({settings.max_cycles}) reached. Stopping.[/yellow]")
                break

    async def _cycle(self) -> None:
        # Filter out completed tasks from previous cycles
        pending_tasks = [t for t in self.tasks if t.status == "pending"]
        
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, pending_tasks)
        
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to our task list
        for task in plan.tasks:
            if not any(t.id == task.id for t in self.tasks):
                self.tasks.append(task)

        # Execute only pending/in-progress tasks
        tasks_to_run = [t for t in self.tasks if t.status in ("pending", "in_progress")]
        
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in tasks_to_run
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(tasks_to_run, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.tasks)

    def _cleanup_state(self) -> None:
        """Remove state file on completion."""
        path = Path(STATE_FILE)
        if path.exists():
            path.unlink()
