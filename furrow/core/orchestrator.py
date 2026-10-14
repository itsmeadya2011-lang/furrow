from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import settings
from furrow.llm import LLMClient

console = Console()

STATE_FILE = ".furrow_state.json"


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[Any] = []
        self._load_state()

    def _load_state(self) -> None:
        path = Path(STATE_FILE)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.cycles = data.get("cycles", 0)
                self.all_tasks = data.get("tasks", [])
                self.original_goal = data.get("original_goal", self.goal)
                console.print(f"[dim]Loaded state: {self.cycles} cycles, {len(self.all_tasks)} tasks[/dim]")
            except json.JSONDecodeError as e:
                console.print(f"[yellow]Failed to load state: {e}[/yellow]")
            except OSError as e:
                console.print(f"[yellow]Failed to load state: {e}[/yellow]")

    def _save_state(self) -> None:
        data = {
            "cycles": self.cycles,
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "files": t.files,
                    "dependencies": t.dependencies,
                    "status": t.status,
                    "result": t.result,
                    "cycle": t.cycle,
                }
                for t in self.all_tasks
            ],
            "original_goal": self.original_goal,
            "current_goal": self.goal,
        }
        Path(STATE_FILE).write_text(json.dumps(data, indent=2))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = settings.max_cycles or 10
        
        while self.cycles < max_cycles:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles}/{max_cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._cleanup_state()
                break
        
        if self.cycles >= max_cycles:
            console.print(f"[yellow]Reached max cycles ({max_cycles}). Stopping.[/yellow]")

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, self.all_tasks, self.cycles)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Filter out already completed tasks
        new_tasks = [t for t in plan.tasks if not self._is_task_completed(t.id)]
        if not new_tasks:
            console.print("[green]All planned tasks already completed.[/green]")
            return

        # Set cycle number for new tasks
        for task in new_tasks:
            task.cycle = self.cycles

        self.all_tasks.extend(new_tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in new_tasks
            ]
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

        with Status("[bold yellow]Testing...", console=console):
            test_result = await self.tester.run(self.original_goal, self.all_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Update goal to include fix instructions but keep original
            self.goal = f"{self.original_goal}\n\nPrevious cycle test failures to fix:\n" + "\n".join(test_result.failures)

        # Run lint checks
        with Status("[bold yellow]Linting...", console=console):
            lint_result = await self.tester.run_lint(self.all_tasks)
        
        if lint_result.passed:
            console.print(f"[green]Lint passed: {lint_result.summary}[/green]")
        else:
            console.print(f"[yellow]Lint issues: {lint_result.summary}[/yellow]")
            for issue in lint_result.issues:
                console.print(f"  • {issue}")
            # Add lint issues to goal for next cycle
            if lint_result.issues:
                self.goal += "\n\nLint issues to fix:\n" + "\n".join(lint_result.issues)

    def _is_task_completed(self, task_id: str) -> bool:
        for t in self.all_tasks:
            if t.id == task_id and t.status == "completed":
                return True
        return False

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False

        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")

        return failed == 0 and completed >= len(self.all_tasks)

    def _cleanup_state(self) -> None:
        path = Path(STATE_FILE)
        if path.exists():
            path.unlink()
