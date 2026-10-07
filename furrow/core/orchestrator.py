from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Awaitable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult, Settings, settings
from furrow.context import compact_context, should_compact
from furrow.git import auto_commit, get_changed_files, is_git_repo
from furrow.llm import LLMClient

console = Console()

# Type for output callback
OutputCallback = Callable[[str], Awaitable[None]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        session_file: Path | None = None,
        output_callback: OutputCallback | None = None,
        auto_commit: bool = True,
        workspace: Path | None = None,
        compact_threshold: int = 10,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient()
        self.settings = settings or Settings()
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.session_file = session_file or Path.cwd() / ".furrow_session.json"
        self.output_callback = output_callback
        self.auto_commit = auto_commit
        self.workspace = workspace or Path.cwd()
        self.compact_threshold = compact_threshold
        self.cycle_history: list[dict] = []
        self._load_session()

    async def _emit(self, message: str) -> None:
        """Emit a message to both console and callback."""
        console.print(message)
        if self.output_callback:
            try:
                await self.output_callback(message)
            except Exception:
                pass  # Don't let callback errors break the loop

    def _load_session(self) -> None:
        if self.session_file.exists():
            try:
                data = json.loads(self.session_file.read_text())
                self.cycles = data.get("cycles", 0)
                self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                self.goal = data.get("goal", self.goal)
                self.original_goal = data.get("original_goal", self.original_goal)
                self.cycle_history = data.get("cycle_history", [])
                console.print(f"[dim]Loaded session: {self.cycles} cycles, {len(self.tasks)} tasks[/dim]")
            except Exception as e:
                console.print(f"[yellow]Could not load session: {e}[/yellow]")

    def _save_session(self) -> None:
        try:
            data = {
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self.tasks],
                "goal": self.goal,
                "original_goal": self.original_goal,
                "cycle_history": self.cycle_history,
            }
            self.session_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            console.print(f"[yellow]Could not save session: {e}[/yellow]")

    async def run(self) -> None:
        await self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_session()

            # Auto-commit changes if enabled and tests passed
            if self.auto_commit and await is_git_repo(self.workspace):
                changed = await get_changed_files(self.workspace)
                if changed:
                    commit_msg = f"furrow: cycle {self.cycles} - {self.goal[:50]}"
                    if await auto_commit(self.workspace, commit_msg):
                        await self._emit(f"[dim]Committed changes: {commit_msg}[/dim]")

            # Check if we should compact context
            if await should_compact(self.cycle_history, self.compact_threshold):
                await self._emit("[dim]Compacting context...[/dim]")
                summary = await compact_context(self.client, self.original_goal, self.cycle_history, self.settings)
                self.goal = f"{self.original_goal}\n\nContext summary:\n{summary}"
                self.cycle_history = []  # Reset history after compaction
                await self._emit("[dim]Context compacted.[/dim]")

            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                self._cleanup_session()
                break

    def _cleanup_session(self) -> None:
        if self.session_file.exists():
            try:
                self.session_file.unlink()
            except Exception:
                pass

    async def _cycle(self) -> None:
        # Filter out already completed tasks
        pending_tasks = [t for t in self.tasks if t.status == "pending"]

        if not pending_tasks:
            await self._emit("[bold yellow]Planning...[/bold yellow]")
            plan = await self.planner.plan(self.goal)
            await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

            if not plan.tasks:
                await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
                return

            # Add new tasks to our task list
            for task in plan.tasks:
                task.status = "pending"
                self.tasks.append(task)
            pending_tasks = plan.tasks

        await self._emit(f"[dim]Executing {len(pending_tasks)} pending task(s)...[/dim]")

        await self._emit("[bold yellow]Executing tasks in parallel...[/bold yellow]")
        worker_tasks = [
            WorkerAgent(task=task, client=self.client, settings=self.settings).run()
            for task in pending_tasks
        ]
        results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        cycle_summary = f"Cycle {self.cycles}: Executed {len(pending_tasks)} tasks"
        completed_count = 0
        failed_count = 0

        for task, result in zip(pending_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
                failed_count += 1
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed: {result}[/green]")
                completed_count += 1

        cycle_summary += f" ({completed_count} completed, {failed_count} failed)"

        await self._emit("[bold yellow]Testing...[/bold yellow]")
        test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, self.tasks)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
            cycle_summary += f" - Tests passed"
            # Check if all tasks are done
            if all(t.status == "completed" for t in self.tasks):
                await self._emit("[bold green]All tasks completed and tests pass![/bold green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            cycle_summary += f" - Tests failed: {test_result.summary}"
            # Append failures to goal instead of replacing
            self.goal = f"{self.original_goal}\n\nFix failing tests:\n" + "\n".join(test_result.failures)

        # Record cycle history
        self.cycle_history.append({
            "cycle": self.cycles,
            "summary": cycle_summary,
            "tasks_completed": completed_count,
            "tasks_failed": failed_count,
            "tests_passed": test_result.passed,
        })

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.tasks)

    def get_status(self) -> dict[str, Any]:
        return {
            "cycles": self.cycles,
            "goal": self.goal,
            "original_goal": self.original_goal,
            "tasks": [t.model_dump() for t in self.tasks],
            "completed": sum(1 for t in self.tasks if t.status == "completed"),
            "failed": sum(1 for t in self.tasks if t.status == "failed"),
            "pending": sum(1 for t in self.tasks if t.status == "pending"),
        }