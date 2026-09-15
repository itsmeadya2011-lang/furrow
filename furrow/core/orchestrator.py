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
from furrow.config import (
    Plan,
    Settings,
    TaskModel,
    TestResult,
    settings as default_settings,
)
from furrow.llm import LLMClient

console = Console()


_TERMINAL_STATUSES = {"completed", "failed"}


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        if isinstance(client, Settings):
            settings = client
            client = None

        self.goal = goal
        self.original_goal = goal
        resolved_settings = settings
        if resolved_settings is None:
            client_settings = getattr(client, "settings", None)
            resolved_settings = (
                client_settings
                if client_settings is not None
                else default_settings
            )
        self.settings = resolved_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self._has_planned = False
        self._tasks_by_id: dict[str, TaskModel] = {}
        self._last_state_signature: tuple[Any, ...] | None = None

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            if self._is_done() and (
                self._has_planned or self.tasks or self.plan is not None
            ):
                self._print_terminal_status()
                break
            if (
                self.settings.max_cycles > 0
                and self.cycles >= self.settings.max_cycles
            ):
                console.print("[yellow]Maximum cycle count reached. Halting.[/yellow]")
                break

            before_signature = self._state_signature()
            self.cycles += 1
            console.print(
                f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]"
            )
            await self._cycle()
            after_signature = self._state_signature()
            self._last_state_signature = after_signature

            if self._is_done():
                self._print_terminal_status()
                break
            if after_signature == before_signature:
                console.print("[yellow]No progress made. Halting.[/yellow]")
                break
            if (
                self.settings.max_cycles > 0
                and self.cycles >= self.settings.max_cycles
            ):
                console.print("[yellow]Maximum cycle count reached. Halting.[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._has_planned = True
        self._merge_plan(plan)
        console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        ready_tasks = self._get_tasks()
        if ready_tasks:
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                await self._execute_tasks(ready_tasks)
        else:
            console.print("[yellow]No dependency-ready tasks to execute.[/yellow]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await self._run_tests()
        self.last_test_result = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")

    def _merge_plan(self, plan: Plan) -> None:
        self._has_planned = True
        self.plan = plan
        self._tasks_by_id = {task.id: task for task in self.tasks}
        for incoming in plan.tasks:
            existing = self._tasks_by_id.get(incoming.id)
            if existing is None:
                self.tasks.append(incoming)
                self._tasks_by_id[incoming.id] = incoming
                continue

            existing.description = incoming.description
            existing.files = incoming.files
            existing.dependencies = incoming.dependencies

        self._mark_blocked_dependents()

    async def _execute_tasks(self, ready_tasks: list[TaskModel]) -> None:
        parallel_limit = max(1, int(self.settings.max_parallel_tasks))
        semaphore = asyncio.Semaphore(parallel_limit)

        async def execute_task(task: TaskModel) -> tuple[TaskModel, Any]:
            async with semaphore:
                task.status = "running"
                try:
                    worker = WorkerAgent(task=task, client=self.client)
                    result = await worker.run()
                except Exception as exc:
                    result = exc
                return task, result

        results = await asyncio.gather(
            *(execute_task(task) for task in ready_tasks),
        )
        for task, result in results:
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = "" if result is None else str(result)
                console.print(f"[green]Task {task.id} completed[/green]")

        self._mark_blocked_dependents()

    async def _run_tests(self) -> TestResult:
        tester = TesterAgent(client=self.client)
        if hasattr(tester, "settings"):
            tester.settings = self.settings
        return await tester.run(
            self.goal,
            list(self.tasks),
        )

    def _sync_plan_tasks(self) -> None:
        if self.plan is None:
            return
        current_ids = {task.id for task in self.tasks}
        for task in self.plan.tasks:
            if task.id not in current_ids:
                self.tasks.append(task)
                current_ids.add(task.id)

    def _mark_blocked_dependents(self) -> None:
        self._sync_plan_tasks()
        self._tasks_by_id = {task.id: task for task in self.tasks}

        changed = True
        while changed:
            changed = False
            for task in self.tasks:
                if task.status not in {"pending", "blocked"}:
                    continue

                missing_dependencies = [
                    dependency
                    for dependency in task.dependencies
                    if dependency not in self._tasks_by_id
                ]
                failed_dependencies = [
                    dependency
                    for dependency in task.dependencies
                    if dependency in self._tasks_by_id
                    and self._tasks_by_id[dependency].status in {"failed", "blocked"}
                ]
                if missing_dependencies or failed_dependencies:
                    blocked_by = missing_dependencies + failed_dependencies
                    result = (
                        "Blocked by unresolved dependencies: "
                        f"{', '.join(blocked_by)}"
                    )
                    if task.status != "blocked" or task.result != result:
                        task.status = "blocked"
                        task.result = result
                        changed = True
                elif task.status == "blocked":
                    task.status = "pending"
                    task.result = None
                    changed = True

    def _blocked_dependency_reason(
        self,
        task: TaskModel,
        seen: set[str] | None = None,
    ) -> str | None:
        if seen is None:
            seen = set()
        if task.id in seen:
            return None
        seen.add(task.id)

        for dependency in task.dependencies:
            dependency_task = self._tasks_by_id.get(dependency)
            if dependency_task is None:
                return "unresolved"
            if dependency_task.status == "failed":
                return "failed"
            if dependency_task.status == "blocked":
                reason = self._blocked_dependency_reason(dependency_task, seen)
                if reason is not None:
                    return reason

        return None

    def _get_ready_tasks(self) -> list[TaskModel]:
        self._mark_blocked_dependents()
        return [
            task
            for task in self.tasks
            if task.status == "pending"
            and all(
                dependency in self._tasks_by_id
                and self._tasks_by_id[dependency].status == "completed"
                for dependency in task.dependencies
            )
        ]

    def _has_task_failures(self) -> bool:
        return any(task.status in {"failed", "blocked"} for task in self.tasks)

    def _has_incomplete_tasks(self) -> bool:
        return any(task.status != "completed" for task in self.tasks)

    def _print_terminal_status(self) -> None:
        if self.last_test_result is not None and not self.last_test_result.passed:
            console.print("[bold red]Tests failed. Halting.[/bold red]")
        elif self._has_task_failures():
            console.print("[bold red]Task failure detected. Halting.[/bold red]")
        elif self._has_incomplete_tasks():
            console.print("[yellow]Work stalled. Halting.[/yellow]")
        else:
            console.print("[bold green]Run complete. Halting.[/bold green]")

    def _state_signature(self) -> tuple[Any, ...]:
        task_signature = tuple(
            (task.id, task.status, tuple(sorted(task.dependencies)))
            for task in sorted(self.tasks, key=lambda task: task.id)
        )
        plan_signature: tuple[Any, ...] = ()
        if self.plan is not None:
            plan_signature = tuple(
                (task.id, tuple(sorted(task.dependencies)))
                for task in sorted(
                    self.plan.tasks,
                    key=lambda task: (task.id, tuple(sorted(task.dependencies))),
                )
            )
        return task_signature, plan_signature

    def _is_done(self) -> bool:
        if not self._has_planned and not self.tasks and self.plan is None:
            return False
        self._mark_blocked_dependents()
        if not self.tasks:
            return True
        if any(task.status == "running" for task in self.tasks):
            return False
        if all(task.status in _TERMINAL_STATUSES for task in self.tasks):
            return True

        blocked_tasks = [task for task in self.tasks if task.status == "blocked"]
        if any(
            self._blocked_dependency_reason(task) == "failed"
            for task in blocked_tasks
        ):
            return True
        if any(
            self._blocked_dependency_reason(task) == "unresolved"
            for task in blocked_tasks
        ):
            return False
        return not self._get_ready_tasks()

    def _get_tasks(self) -> list[TaskModel]:
        self._mark_blocked_dependents()
        parallel_limit = max(1, int(self.settings.max_parallel_tasks))
        return self._get_ready_tasks()[:parallel_limit]
