from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import FileChange, Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            settings = self.client.settings
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                console.print(
                    f"[yellow]Max cycles ({settings.max_cycles}) reached. Halting.[/yellow]"
                )
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge newly planned tasks into the tracked task list, avoiding duplicates by id.
        existing_ids = {t.id for t in self.all_tasks}
        new_tasks = [t for t in plan.tasks if t.id not in existing_ids]
        self.all_tasks.extend(new_tasks)

        # Execute tasks respecting dependencies (topological order).
        results = await self._execute_tasks(new_tasks)

        for task, result in zip(new_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result.summary if hasattr(result, "summary") else str(result)
                console.print(f"[green]Task {task.id} completed[/green]")
                # Apply file changes to disk.
                if hasattr(result, "files"):
                    for file_change in result.files:
                        self._apply_file_change(file_change)
                        console.print(f"  [dim]wrote {file_change.path}[/dim]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, new_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[TaskModel]) -> list[Any]:
        """Execute tasks respecting dependency ordering.

        Tasks with no dependencies run in parallel. Tasks that depend on others
        wait until their dependencies complete.
        """
        if not tasks:
            return []

        # Build adjacency: task_id -> list of task_ids it depends on
        task_map = {t.id: t for t in tasks}
        pending = set(t.id for t in tasks)
        completed: set[str] = set()
        results: dict[str, Any] = {}

        async def run_one(task: TaskModel) -> Any:
            worker = WorkerAgent(task=task, client=self.client)
            return await worker.run()

        while pending:
            # Find tasks whose dependencies are all completed.
            ready = [
                task_map[tid]
                for tid in list(pending)
                if all(dep in completed for dep in task_map[tid].dependencies)
            ]
            if not ready:
                # Circular dependency or unresolvable: run remaining in parallel anyway.
                ready = [task_map[tid] for tid in list(pending)]

            batch_results = await asyncio.gather(
                *[run_one(t) for t in ready], return_exceptions=True
            )
            for task, result in zip(ready, batch_results):
                results[task.id] = result
                completed.add(task.id)
                pending.discard(task.id)

        return [results[t.id] for t in tasks]

    def _apply_file_change(self, file_change: FileChange) -> None:
        p = Path(file_change.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(file_change.content, encoding="utf-8")

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.all_tasks)

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks
