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
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()


def _run_git(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Git command timed out"
    except Exception as e:
        return -1, "", str(e)


def _is_git_repo(path: Path) -> bool:
    """Check if path is a git repository."""
    code, _, _ = _run_git(["git", "rev-parse", "--git-dir"], path)
    return code == 0


def _git_commit(cwd: Path, message: str) -> bool:
    """Stage all changes and commit with message."""
    code, _, _ = _run_git(["git", "add", "-A"], cwd)
    if code != 0:
        return False
    code, _, _ = _run_git(["git", "commit", "-m", message], cwd)
    return code == 0


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings_obj: Settings | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.settings = settings_obj or settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.task_history: list[TaskModel] = []
        self.state_file = self.settings.workspace / ".furrow_state.json"
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self.task_history = [TaskModel(**t) for t in data.get("task_history", [])]
                console.print(f"[dim]Loaded state: {len(self.task_history)} tasks from {self.cycles} previous cycles[/dim]")
            except Exception as e:
                console.print(f"[yellow]Failed to load state: {e}[/yellow]")

    def _save_state(self) -> None:
        data = {
            "goal": self.goal,
            "cycles": self.cycles,
            "task_history": [t.model_dump() for t in self.task_history],
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            
            # Check max_cycles limit
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Stopping.[/yellow]")
                break
                
            await self._cycle()
            self._save_state()
            
            # Auto-commit if in a git repo
            if _is_git_repo(self.settings.workspace):
                commit_msg = f"furrow: cycle {self.cycles} - {self._summarize_cycle()}"
                if _git_commit(self.settings.workspace, commit_msg):
                    console.print(f"[dim]Committed: {commit_msg}[/dim]")
            
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    def _summarize_cycle(self) -> str:
        """Create a short summary of what was done this cycle."""
        completed = [t for t in self.task_history if t.status == "completed"]
        if completed:
            return f"completed {len(completed)} tasks"
        return "no changes"

    async def _cycle(self) -> None:
        # Provide context from previous cycles to planner
        context = self._build_context()
        
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, context)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Filter out already completed tasks
        pending_tasks = [t for t in plan.tasks if t.id not in {h.id for h in self.task_history if h.status == "completed"}]
        
        if not pending_tasks:
            console.print("[green]All planned tasks already completed.[/green]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in pending_tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(pending_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
            
            # Update task history
            self._update_task_history(task)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, pending_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Append failures to goal instead of replacing
            self.goal = f"{self.goal}\n\nPrevious test failures to fix:\n" + "\n".join(f"- {f}" for f in test_result.failures)

    def _build_context(self) -> str:
        if not self.task_history:
            return ""
        
        completed = [t for t in self.task_history if t.status == "completed"]
        failed = [t for t in self.task_history if t.status == "failed"]
        
        parts = ["Previous cycle context:"]
        if completed:
            parts.append("Completed tasks:")
            for t in completed[-5:]:  # Last 5 completed
                parts.append(f"  - {t.id}: {t.description[:80]}")
        if failed:
            parts.append("Failed tasks:")
            for t in failed[-3:]:  # Last 3 failed
                parts.append(f"  - {t.id}: {t.description[:80]} (error: {t.result[:100] if t.result else 'unknown'})")
        return "\n".join(parts)

    def _update_task_history(self, task: TaskModel) -> None:
        # Remove existing entry with same id
        self.task_history = [t for t in self.task_history if t.id != task.id]
        self.task_history.append(task)

    def _is_done(self) -> bool:
        if not self.task_history:
            return False
        
        completed = sum(1 for t in self.task_history if t.status == "completed")
        failed = sum(1 for t in self.task_history if t.status == "failed")
        total_planned = len(self.task_history)
        
        if failed > 0:
            return False
        if completed >= total_planned and total_planned > 0:
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.task_history

    def print_status(self) -> None:
        """Print a summary table of all tasks."""
        table = Table(title="Task Status")
        table.add_column("ID", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="bold")
        table.add_column("Result", style="dim")
        
        for task in self.task_history:
            status_style = {
                "completed": "[green]✓ completed[/green]",
                "failed": "[red]✗ failed[/red]",
                "pending": "[yellow]⏳ pending[/yellow]",
            }.get(task.status, task.status)
            
            result = task.result[:80] + "..." if task.result and len(task.result) > 80 else task.result or ""
            table.add_row(task.id, task.description[:60], status_style, result)
        
        console.print(table)
