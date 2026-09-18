from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings as module_settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.current_goal = goal
        self.settings = settings
        if self.settings is None:
            self.settings = getattr(client, "settings", None) or module_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings)
        self.tasks: list[TaskModel] = []
        self.cycles = 0

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}", title="Furrow"))

        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(
                    f"[yellow]Cycle limit reached ({self.settings.max_cycles}). Halting.[/yellow]"
                )
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            try:
                plan = await self.planner.plan(self.current_goal)
            except Exception as exc:
                console.print(f"[red]Planning failed: {exc}[/red]")
                break

            console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
            self._merge_plan(plan)
            self._print_task_states()

            if not plan.tasks:
                console.print("[yellow]No tasks planned. Halting.[/yellow]")
                break

            await self._execute_pending_tasks()

            with Status("[bold yellow]Testing...", console=console):
                test_result = await self.tester.run(self.current_goal, self.tasks)
            self._print_test_result(test_result)

            if test_result.passed:
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

            if not self._has_runnable_tasks():
                console.print(
                    "[red]Tests failed and no runnable tasks remain. Halting.[/red]"
                )
                break

            failures = "\n".join(f"- {failure}" for failure in test_result.failures)
            self.current_goal = (
                "Fix failing tests while preserving the original goal.\n\n"
                f"Original goal:\n{self.original_goal}\n\n"
                f"Test failures:\n{failures}"
            )

    def _merge_plan(self, plan: Plan) -> None:
        existing = {task.id: task for task in self.tasks}
        for incoming in plan.tasks:
            current = existing.get(incoming.id)
            if current is None:
                self.tasks.append(incoming)
                existing[incoming.id] = incoming
                continue

            current.description = incoming.description
            current.files = list(incoming.files)
            current.dependencies = list(incoming.dependencies)
            if incoming.status == "pending" and "status" in incoming.model_fields_set:
                current.status = "pending"
                current.result = None

    async def _execute_pending_tasks(self) -> None:
        semaphore = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))

        while True:
            ready: list[TaskModel] = []
            blocked: list[tuple[TaskModel, str]] = []
            by_id = {task.id: task for task in self.tasks}

            for task in self.tasks:
                if task.status != "pending":
                    continue

                dependency_states = [
                    (dependency, by_id.get(dependency)) for dependency in task.dependencies
                ]
                unavailable = [
                    dependency
                    for dependency, dependency_task in dependency_states
                    if dependency_task is None or dependency_task.status in {"failed", "blocked"}
                ]
                if unavailable:
                    reason = "missing or unavailable dependencies: " + ", ".join(unavailable)
                    blocked.append((task, reason))
                    continue
                if all(dependency_task.status == "completed" for _, dependency_task in dependency_states):
                    ready.append(task)

            for task, reason in blocked:
                task.status = "blocked"
                task.result = reason
                console.print(f"[red]Task {task.id} blocked: {reason}[/red]")

            if not ready:
                pending = [task.id for task in self.tasks if task.status == "pending"]
                if pending:
                    console.print(
                        "[yellow]Pending tasks are waiting for unfinished dependencies: "
                        + ", ".join(pending)
                        + "[/yellow]"
                    )
                break

            async def execute_task(task: TaskModel) -> None:
                async with semaphore:
                    task.status = "running"
                    console.print(f"[cyan]Task {task.id} running[/cyan]")
                    try:
                        result = await WorkerAgent(
                            task=task,
                            client=self.client,
                            settings=self.settings,
                        ).run()
                    except Exception as exc:
                        task.status = "failed"
                        task.result = str(exc)
                        console.print(f"[red]Task {task.id} failed: {exc}[/red]")
                    else:
                        task.status = "completed"
                        task.result = result if isinstance(result, str) else str(result)
                        console.print(f"[green]Task {task.id} completed[/green]")

            await asyncio.gather(*(execute_task(task) for task in ready))
            self._print_task_states()

    def _has_runnable_tasks(self) -> bool:
        by_id = {task.id: task for task in self.tasks}
        for task in self.tasks:
            if task.status != "pending":
                continue
            dependencies = [by_id.get(dependency) for dependency in task.dependencies]
            if all(dependency is not None and dependency.status == "completed" for dependency in dependencies):
                return True
        return False

    def _print_task_states(self) -> None:
        table = Table(title="Task states", border_style="blue")
        table.add_column("ID", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Description")
        table.add_column("Dependencies")
        for task in self.tasks:
            table.add_row(
                task.id,
                task.status,
                task.description,
                ", ".join(task.dependencies) or "-",
            )
        console.print(table)

    def _print_test_result(self, test_result: TestResult) -> None:
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            return

        console.print(f"[red]Tests failed: {test_result.summary}[/red]")
        for failure in test_result.failures:
            console.print(f"  [red]• {failure}[/red]")
        console.print("[yellow]Will attempt fix in the next cycle.[/yellow]")

    def _is_done(self) -> bool:
        return all(task.status == "completed" for task in self._get_tasks())

    def _get_tasks(self) -> list[Any]:
        return self.tasks
