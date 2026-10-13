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
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, TaskModel, settings
from furrow.llm import LLMClient

console = Console()

OutputCallback = Callable[[str], Awaitable[None]] | None


class Orchestrator:
    def __init__(
        self, 
        goal: str, 
        client: LLMClient | None = None, 
        state_file: Path | None = None,
        output_callback: OutputCallback = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.state_file = state_file or (settings.workspace / ".furrow_state.json")
        self.output_callback = output_callback
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                self._output(f"[dim]Loaded state: {len(self.tasks)} tasks, cycle {self.cycles}[/dim]")
            except Exception as e:
                self._output(f"[yellow]Failed to load state: {e}[/yellow]")

    def _save_state(self) -> None:
        try:
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self.tasks],
            }
            self.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            self._output(f"[yellow]Failed to save state: {e}[/yellow]")

    async def _output(self, message: str) -> None:
        console.print(message)
        if self.output_callback:
            await self.output_callback(message)

    async def run(self) -> None:
        await self._output(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = settings.max_cycles if settings.max_cycles > 0 else float('inf')
        while self.cycles < max_cycles:
            self.cycles += 1
            await self._output(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                await self._output("[bold green]Goal complete. Halting.[/bold green]")
                break
        else:
            await self._output(f"[yellow]Max cycles ({max_cycles}) reached. Stopping.[/yellow]")

    async def _cycle(self) -> None:
        # Build context from previous cycles
        context = self._build_context()
        
        await self._output("[bold yellow]Planning...[/bold yellow]")
        plan = await self.planner.plan(self.goal, context=context)
        await self._output(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._output("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge new tasks with existing (avoid duplicates by id)
        for new_task in plan.tasks:
            existing = next((t for t in self.tasks if t.id == new_task.id), None)
            if existing:
                # Update existing task with new plan info
                existing.description = new_task.description
                existing.files = new_task.files
                existing.dependencies = new_task.dependencies
                existing.status = "pending"
                existing.result = None
            else:
                self.tasks.append(new_task)

        # Filter to only pending tasks for this cycle
        pending_tasks = [t for t in self.tasks if t.status == "pending"]
        
        # Execute in parallel with max_parallel_tasks limit
        semaphore = asyncio.Semaphore(settings.max_parallel_tasks)
        
        async def run_with_semaphore(task: TaskModel) -> tuple[TaskModel, Any]:
            async with semaphore:
                result = await WorkerAgent(task=task, client=self.client).run()
                return task, result

        await self._output("[bold yellow]Executing tasks in parallel...[/bold yellow]")
        task_futures = [run_with_semaphore(task) for task in pending_tasks]
        results = await asyncio.gather(*task_futures, return_exceptions=True)

        for task, result in results:
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._output(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._output(f"[green]Task {task.id} completed[/green]")

        await self._output(self._get_task_table_string())

        await self._output("[bold yellow]Testing...[/bold yellow]")
        test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        if test_result.passed:
            await self._output(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._output(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._output(f"  • {failure}")
            await self._output("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Add fix tasks for failures instead of overwriting goal
            self._add_fix_tasks(test_result.failures)

    def _build_context(self) -> str:
        if not self.tasks:
            return ""
        completed = [t for t in self.tasks if t.status == "completed"]
        failed = [t for t in self.tasks if t.status == "failed"]
        pending = [t for t in self.tasks if t.status == "pending"]
        
        parts = ["## Previous Progress"]
        if completed:
            parts.append("### Completed Tasks:")
            for t in completed:
                parts.append(f"  - {t.id}: {t.description} → {t.result[:100] if t.result else 'done'}")
        if failed:
            parts.append("### Failed Tasks:")
            for t in failed:
                parts.append(f"  - {t.id}: {t.description} → ERROR: {t.result[:100] if t.result else 'unknown'}")
        if pending:
            parts.append("### Pending Tasks:")
            for t in pending:
                parts.append(f"  - {t.id}: {t.description}")
        return "\n".join(parts)

    def _add_fix_tasks(self, failures: list[str]) -> None:
        for i, failure in enumerate(failures):
            fix_id = f"fix_{self.cycles}_{i}"
            if not any(t.id == fix_id for t in self.tasks):
                self.tasks.append(TaskModel(
                    id=fix_id,
                    description=f"Fix: {failure}",
                    files=[],
                    dependencies=[],
                    status="pending",
                ))

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        pending = sum(1 for t in self.tasks if t.status == "pending")
        if failed > 0:
            return False
        if pending == 0 and completed > 0:
            return True
        return False

    def _print_task_table(self) -> None:
        table = Table(title="Task Status")
        table.add_column("ID", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="bold")
        table.add_column("Result", style="dim")
        for task in self.tasks:
            status_style = {
                "completed": "green",
                "failed": "red",
                "pending": "yellow",
            }.get(task.status, "white")
            result_preview = (task.result[:80] + "...") if task.result and len(task.result) > 80 else (task.result or "")
            table.add_row(task.id, task.description, f"[{status_style}]{task.status}[/{status_style}]", result_preview)
        console.print(table)

    def _get_task_table_string(self) -> str:
        from io import StringIO
        from rich.console import Console as RichConsole
        
        string_io = StringIO()
        temp_console = RichConsole(file=string_io, force_terminal=True, width=120)
        
        table = Table(title="Task Status")
        table.add_column("ID", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="bold")
        table.add_column("Result", style="dim")
        for task in self.tasks:
            status_style = {
                "completed": "green",
                "failed": "red",
                "pending": "yellow",
            }.get(task.status, "white")
            result_preview = (task.result[:80] + "...") if task.result and len(task.result) > 80 else (task.result or "")
            table.add_row(task.id, task.description, f"[{status_style}]{task.status}[/{status_style}]", result_preview)
        temp_console.print(table)
        
        return string_io.getvalue()
