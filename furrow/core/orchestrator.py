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
from furrow.config import Settings, TaskModel, settings
from furrow.llm import LLMClient


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        orchestrator_settings: Settings | None = None,
        console: Console | None = None,
        state_file: str | Path | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings_obj=orchestrator_settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[Any] = []
        self.settings = orchestrator_settings or settings
        self.console = console or Console()
        self.state_file = Path(state_file) if state_file else None
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file and self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                tasks_data = data.get("tasks", [])
                self._tasks = [TaskModel(**t) for t in tasks_data]
                self.console.print(f"[cyan]Loaded state from {self.state_file} (cycle {self.cycles})[/cyan]")
            except Exception as e:  # noqa: BLE001
                self.console.print(f"[yellow]Failed to load state: {e}[/yellow]")

    def _save_state(self) -> None:
        if not self.state_file:
            return
        try:
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self._tasks],
            }
            self.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:  # noqa: BLE001
            self.console.print(f"[yellow]Failed to save state: {e}[/yellow]")

    async def run(self) -> None:
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                self.console.print(f"[bold red]Max cycles ({self.settings.max_cycles}) reached. Halting.[/bold red]")
                break

            await self._cycle()
            self._save_state()

            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console):
            plan = await self.planner.plan(self.goal, self._tasks)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self._tasks = plan.tasks

        with Status("[bold yellow]Executing tasks with dependencies...", console=self.console):
            await self._execute_tasks_with_dependencies(plan.tasks)

        with Status("[bold yellow]Testing...", console=self.console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self._tasks:
            return True

        completed = sum(1 for t in self._tasks if t.status == "completed")
        failed = sum(1 for t in self._tasks if t.status == "failed")

        if failed > 0:
            return False
        return completed >= len(self._tasks)

    def _get_tasks(self) -> list[Any]:
        return self._tasks

    async def _execute_tasks_with_dependencies(self, tasks: list[TaskModel]) -> None:
        task_map = {t.id: t for t in tasks}
        completed = set()
        pending = {t.id: t for t in tasks}
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def run_task(task: TaskModel) -> tuple[str, str | Exception | BaseException]:
            async with semaphore:
                self.console.print(f"[cyan]Starting task {task.id}: {task.description[:50]}...[/cyan]")
                try:
                    result = await WorkerAgent(task=task, client=self.client).run()
                    return task.id, result
                except Exception as e:  # noqa: BLE001
                    return task.id, e

        while pending:
            ready = [
                task for task in pending.values()
                if all(dep in completed for dep in task.dependencies)
            ]

            if not ready:
                remaining = list(pending.values())
                self.console.print(f"[red]Circular dependency detected or missing dependencies: {[t.id for t in remaining]}[/red]")
                for task in remaining:
                    task.status = "failed"
                    task.result = "Circular dependency or missing dependency"
                break

            running_tasks = [run_task(task) for task in ready]
            for task in ready:
                del pending[task.id]

            results: list[tuple[str, str | Exception | BaseException] | BaseException] = await asyncio.gather(
                *running_tasks, return_exceptions=True
            )

            for item in results:
                if isinstance(item, BaseException):
                    # This shouldn't happen with our run_task, but handle it
                    self.console.print(f"[red]Task execution failed with: {item}[/red]")
                    continue
                task_id, result = item
                task = task_map[task_id]
                if isinstance(result, BaseException):
                    task.status = "failed"
                    task.result = str(result)
                    self.console.print(f"[red]Task {task_id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    self.console.print(f"[green]Task {task_id} completed[/green]")
                completed.add(task_id)