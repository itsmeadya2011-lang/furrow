from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from typing_extensions import Self

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import settings
from furrow.context import ContextManager
from furrow.git import GitManager
from furrow.llm import LLMClient

console = Console()


class OutputHandler:
    """Abstract output handler for both console and web streaming."""

    def __init__(self, websocket: Any | None = None):
        self.websocket = websocket

    async def print(self, message: str) -> None:
        if self.websocket:
            await self.websocket.send_text(message)
        else:
            console.print(message)

    async def print_panel(self, content: str, title: str, border_style: str = "blue") -> None:
        if self.websocket:
            await self.websocket.send_text(f"[{title}] {content}")
        else:
            console.print(Panel(content, title=title, border_style=border_style))

    def status(self, message: str) -> AsyncStatusContext:
        return AsyncStatusContext(self, message)


class AsyncStatusContext:
    def __init__(self, handler: OutputHandler, message: str):
        self.handler = handler
        self.message = message

    async def __aenter__(self) -> Self:
        await self.handler.print(f"[bold yellow]{self.message}[/bold yellow]")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_handler: OutputHandler | None = None,
        max_cycles: int | None = None,
        git_manager: GitManager | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = max_cycles or settings.max_cycles or 0  # 0 = unlimited
        self.tasks: list[Any] = []  # Track all tasks across cycles
        self.output = output_handler or OutputHandler()
        self._cycle_tasks: list[Any] = []  # Tasks from current cycle
        self.git = git_manager or GitManager(Path(settings.workspace))
        self._git_initialized = False
        self.context = ContextManager(Path(settings.workspace), self.client)

    async def run(self) -> None:
        # Initialize git and create initial checkpoint
        await self._init_git()

        await self.output.print_panel(
            f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
            title="Furrow",
            border_style="green",
        )
        while True:
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                await self.output.print("[yellow]Max cycles reached. Halting.[/yellow]")
                break

            self.cycles += 1
            await self.output.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                await self.output.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _init_git(self) -> None:
        if self._git_initialized:
            return
        if await self.git.is_repo():
            await self.git.get_initial_commit()
            await self.output.print("[green]Git repository detected. Tracking changes.[/green]")
        else:
            await self.git.init()
            await self.output.print("[yellow]Initialized new git repository.[/yellow]")
        self._git_initialized = True

    async def _cycle(self) -> None:
        # Get context for planner
        planner_context = self.context.get_context_for_planner()
        goal_with_context = f"{self.goal}\n\n{planner_context}" if planner_context else self.goal

        async with self.output.status("Planning..."):
            plan = await self.planner.plan(goal_with_context)

        await self.output.print_panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")

        if not plan.tasks:
            await self.output.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self._cycle_tasks = plan.tasks

        # Execute tasks (including any fix tasks from previous cycle)
        changes = await self._execute_tasks(plan.tasks)
        self.tasks.extend(plan.tasks)

        # Update context with changes
        if changes:
            await self.context.update_from_changes(changes, self.cycles)

        # Test and fix loop
        max_fix_attempts = 3
        for fix_attempt in range(max_fix_attempts):
            async with self.output.status("Testing..."):
                test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

            if test_result.passed:
                await self.output.print(f"[green]Tests passed: {test_result.summary}[/green]")
                break

            await self.output.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self.output.print(f"  • {failure}")

            if fix_attempt < max_fix_attempts - 1:
                await self.output.print(
                    f"[yellow]Generating fix tasks (attempt {fix_attempt + 1}/{max_fix_attempts})...[/yellow]"
                )
                fix_tasks = await TesterAgent(client=self.client).fix_failures(
                    self.goal, test_result.failures, plan.tasks
                )
                if fix_tasks:
                    await self.output.print(f"[cyan]Created {len(fix_tasks)} fix tasks[/cyan]")
                    fix_changes = await self._execute_tasks(fix_tasks)
                    self.tasks.extend(fix_tasks)
                    if fix_changes:
                        await self.context.update_from_changes(fix_changes, self.cycles)
                else:
                    await self.output.print(
                        "[yellow]Could not generate fix tasks, updating goal[/yellow]"
                    )
                    self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
                    break
            else:
                await self.output.print(
                    "[red]Max fix attempts reached. Updating goal for next cycle.[/red]"
                )
                self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        # Add cycle summary to context
        completed_count = sum(1 for t in plan.tasks if t.status == "completed")
        self.context.add_cycle_summary(
            f"Cycle {self.cycles}: {completed_count}/{len(plan.tasks)} tasks completed. "
            f"Tests: {'passed' if test_result.passed else 'failed'}"
        )

        # Create git checkpoint after each cycle
        if self._git_initialized:
            checkpoint = await self.git.save_checkpoint(self.cycles, self.goal)
            if checkpoint:
                await self.output.print(f"[dim]Checkpoint saved: {checkpoint[:8]}[/dim]")

    async def _execute_tasks(self, tasks: list[Any]) -> list[dict]:
        """Execute a list of tasks in parallel. Returns list of changes made."""
        if not tasks:
            return []
        all_changes = []
        async with self.output.status("Executing tasks in parallel..."):
            worker_tasks = [
                WorkerAgent(task=task, client=self.client, context=self.context).run()
                for task in tasks
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self.output.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self.output.print(f"[green]Task {task.id} completed[/green]")
                # Parse changes from result if it's JSON
                try:
                    import json

                    data = json.loads(result)
                    if "changes" in data:
                        all_changes.extend(data["changes"])
                except (json.JSONDecodeError, ValueError):
                    pass

        return all_changes

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.tasks)

    def get_task_summary(self) -> dict[str, int]:
        """Get summary of task statuses."""
        summary = {"pending": 0, "completed": 0, "failed": 0}
        for task in self.tasks:
            summary[task.status] = summary.get(task.status, 0) + 1
        return summary
