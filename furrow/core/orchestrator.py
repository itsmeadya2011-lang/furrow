from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import OrchestratorState, TaskModel, settings
from furrow.llm import LLMClient

console = Console()


def _run_git(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)


def is_git_repo(path: Path) -> bool:
    """Check if path is a git repository."""
    code, _, _ = _run_git(["git", "rev-parse", "--git-dir"], path)
    return code == 0


def git_status(path: Path) -> str:
    """Get git status short format."""
    code, out, _ = _run_git(["git", "status", "--short"], path)
    return out if code == 0 else ""


def git_diff(path: Path) -> str:
    """Get git diff of staged and unstaged changes."""
    code, out, _ = _run_git(["git", "diff", "HEAD"], path)
    return out if code == 0 else ""


def git_commit(path: Path, message: str) -> bool:
    """Commit all changes with message."""
    code, _, err = _run_git(["git", "add", "-A"], path)
    if code != 0:
        return False
    code, _, err = _run_git(["git", "commit", "-m", message], path)
    return code == 0


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int = 0,
        state_file: Path | None = None,
        resume: bool = False,
        auto_commit: bool = True,
    ) -> None:
        self.client = client or LLMClient()
        self.max_cycles = max_cycles or settings.max_cycles
        self.state_file = state_file or settings.state_file
        self.auto_commit = auto_commit
        self.workspace = settings.workspace
        self.planner = PlannerAgent(client=self.client)
        
        if resume and self.state_file.exists():
            self._load_state()
        else:
            self.goal = goal
            self.cycles = 0
            self.all_tasks: list[TaskModel] = []

    def _load_state(self) -> None:
        try:
            with open(self.state_file) as f:
                data = json.load(f)
            state = OrchestratorState(**data)
            self.goal = state.goal
            self.cycles = state.cycles
            self.all_tasks = state.tasks
            console.print(f"[green]Resumed from {self.state_file} (cycle {self.cycles})[/green]")
        except Exception as e:
            console.print(f"[yellow]Failed to load state: {e}. Starting fresh.[/yellow]")
            self.goal = ""
            self.cycles = 0
            self.all_tasks = []

    def _save_state(self) -> None:
        state = OrchestratorState(goal=self.goal, cycles=self.cycles, tasks=self.all_tasks)
        try:
            with open(self.state_file, "w") as f:
                json.dump(state.model_dump(), f, indent=2)
        except Exception as e:
            console.print(f"[yellow]Failed to save state: {e}[/yellow]")

    def _maybe_commit(self, cycle: int) -> None:
        if not self.auto_commit:
            return
        if not is_git_repo(self.workspace):
            return
        status = git_status(self.workspace)
        if not status.strip():
            return
        diff = git_diff(self.workspace)
        if diff.strip():
            msg = f"furrow: cycle {cycle} - {self.goal[:50]}"
            if git_commit(self.workspace, msg):
                console.print(f"[green]Committed changes: {msg}[/green]")
            else:
                console.print("[yellow]Failed to commit changes[/yellow]")

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.max_cycles and self.cycles >= self.max_cycles:
                console.print("[yellow]Max cycles reached. Halting.[/yellow]")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            self._maybe_commit(self.cycles)
            # Show token usage
            usage = self.client.get_token_usage()
            if usage["total_tokens"] > 0:
                console.print(f"[dim]Tokens: {usage['input_tokens']} in, {usage['output_tokens']} out, {usage['total_tokens']} total[/dim]")
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to our tracking list
        for task in plan.tasks:
            # Check if task already exists (by id)
            existing = next((t for t in self.all_tasks if t.id == task.id), None)
            if existing:
                # Update existing task with new plan info
                existing.description = task.description
                existing.files = task.files
                existing.dependencies = task.dependencies
            else:
                self.all_tasks.append(task)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
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

        self._print_progress()

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

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

    def _print_progress(self) -> None:
        table = Table(title="Task Progress")
        table.add_column("ID", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="bold")
        table.add_column("Files", style="dim")
        
        for task in self.all_tasks:
            status_style = {
                "completed": "[green]✓ completed[/green]",
                "failed": "[red]✗ failed[/red]",
                "pending": "[yellow]⏳ pending[/yellow]",
            }.get(task.status, task.status)
            table.add_row(task.id, task.description[:60], status_style, ", ".join(task.files[:3]) if task.files else "—")
        
        console.print(table)
