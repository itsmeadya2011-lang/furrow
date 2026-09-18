from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client if client is not None else LLMClient()
        client_settings = getattr(client, "settings", None)
        self.effective_settings = (
            client_settings if client_settings is not None else settings
        )
        self.settings = self.effective_settings
        self.planner = PlannerAgent(client=self.client)
        self.tasks: dict[str, TaskModel] = {}
        self._last_test_result: TestResult | None = None
        self._cycle_progress = False
        self.cycles = 0

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            progress = await self._cycle()

            if not progress:
                if (
                    self._last_test_result is not None
                    and not self._last_test_result.passed
                ):
                    console.print(
                        "[bold red]Tests remain failing and no progress was made. "
                        "Halting.[/bold red]"
                    )
                    break
                if any(task.status == "pending" for task in self._get_tasks()):
                    console.print(
                        "[yellow]No progress made. Pending tasks remain. Halting.[/yellow]"
                    )
                    break

            if any(task.status == "failed" for task in self._get_tasks()):
                console.print("[bold red]Task failures detected. Halting.[/bold red]")
                break
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if (
                self.effective_settings.max_cycles > 0
                and self.cycles >= self.effective_settings.max_cycles
            ):
                console.print(
                    "[yellow]Maximum cycles "
                    f"({self.effective_settings.max_cycles}) reached. Halting.[/yellow]"
                )
                break

    async def _cycle(self) -> bool:
        self._cycle_progress = False
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(
            Panel(
                Pretty(plan.model_dump()),
                title="Plan",
                border_style="blue",
            )
        )

        if plan.tasks:
            self._cycle_progress = self._merge_tasks(plan.tasks)
        self._cycle_progress = self._mark_blocked_tasks() or self._cycle_progress
        ready_tasks = self._get_ready_tasks()
        work_ran = bool(ready_tasks)

        if ready_tasks:
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                results = await self._execute_tasks(ready_tasks)

            for task, result in zip(ready_tasks, results):
                if isinstance(result, BaseException):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
            self._cycle_progress = True

        self._cycle_progress = self._mark_blocked_tasks() or self._cycle_progress

        if work_ran:
            with Status("[bold yellow]Testing...", console=console):
                test_result = await TesterAgent(client=self.client).run(
                    self.goal, self._get_tasks()
                )
            self._last_test_result = test_result

            if test_result.passed:
                console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            else:
                console.print(f"[red]Tests failed: {test_result.summary}[/red]")
                for failure in test_result.failures:
                    console.print(f"  • {failure}")
                console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
                failure_details = (
                    "\n".join(test_result.failures)
                    or test_result.summary
                    or "No failure details were provided."
                )
                self.goal = (
                    "Original goal: "
                    f"{self.original_goal}\n\nFix failing tests:\n{failure_details}"
                )

        return self._cycle_progress

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if any(task.status == "failed" for task in tasks):
            return False
        if any(task.status == "pending" for task in tasks):
            return False
        if self._last_test_result is not None and not self._last_test_result.passed:
            return False
        return True

    def _get_tasks(self) -> list[TaskModel]:
        return list(self.tasks.values())

    def _merge_tasks(self, planned_tasks: list[TaskModel]) -> bool:
        changed = False
        for task in planned_tasks:
            existing = self.tasks.get(task.id)
            if existing is None:
                self.tasks[task.id] = task
                changed = True
                continue

            if existing.status in {"completed", "failed"}:
                continue

            planned_values = task.model_dump(exclude={"status", "result"})
            existing_values = existing.model_dump(exclude={"status", "result"})
            if planned_values == existing_values:
                continue

            existing.description = task.description
            existing.files = list(task.files)
            existing.dependencies = list(task.dependencies)
            changed = True
        return changed

    def _get_ready_tasks(self) -> list[TaskModel]:
        return [
            task
            for task in self.tasks.values()
            if task.status == "pending"
            and all(
                dependency in self.tasks
                and self.tasks[dependency].status == "completed"
                for dependency in task.dependencies
            )
        ]

    def _find_unresolvable_tasks(self, tasks: list[TaskModel]) -> set[str]:
        by_id = {task.id: task for task in tasks}
        remaining = set(by_id)
        while remaining:
            ready = {
                task_id
                for task_id in remaining
                if all(
                    dependency not in remaining
                    for dependency in by_id[task_id].dependencies
                )
            }
            if not ready:
                break
            remaining -= ready
        return remaining

    def _fail_task(self, task: TaskModel, reason: str) -> None:
        task.status = "failed"
        task.result = reason
        console.print(f"[red]Task {task.id} blocked: {reason}[/red]")

    def _mark_blocked_tasks(self) -> bool:
        changed = False
        while True:
            iteration_changed = False
            for task in self.tasks.values():
                if task.status != "pending":
                    continue

                missing = [
                    dependency
                    for dependency in task.dependencies
                    if dependency not in self.tasks
                ]
                blocked = [
                    dependency
                    for dependency in task.dependencies
                    if dependency in self.tasks
                    and self.tasks[dependency].status not in {"completed", "pending"}
                ]
                if missing or blocked:
                    reasons = []
                    if missing:
                        reasons.append(f"missing dependencies: {', '.join(missing)}")
                    if blocked:
                        reasons.append(f"blocked dependencies: {', '.join(blocked)}")
                    self._fail_task(task, "; ".join(reasons))
                    iteration_changed = True

            pending = [task for task in self.tasks.values() if task.status == "pending"]
            for task_id in self._find_unresolvable_tasks(pending):
                task = self.tasks[task_id]
                if task.status == "pending":
                    self._fail_task(
                        task,
                        "cyclic or unresolvable dependencies: "
                        + ", ".join(sorted(task.dependencies)),
                    )
                    iteration_changed = True

            if not iteration_changed:
                break
            changed = True
        return changed
    async def _execute_tasks(self, tasks: list[TaskModel]) -> list[Any]:
        semaphore = (
            asyncio.Semaphore(self.effective_settings.max_parallel_tasks)
            if self.effective_settings.max_parallel_tasks > 0
            else None
        )

        async def execute(task: TaskModel) -> Any:
            if semaphore is None:
                return await WorkerAgent(task=task, client=self.client).run()
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        return await asyncio.gather(
            *(execute(task) for task in tasks), return_exceptions=True
        )
