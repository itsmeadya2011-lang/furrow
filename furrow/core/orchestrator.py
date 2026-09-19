from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings as default_settings
from furrow.llm import LLMClient

console = Console()


class RunStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    CYCLE_LIMIT = "cycle_limit"


@dataclass(frozen=True)
class RunResult:
    status: RunStatus
    cycles: int
    message: str = ""
    test_result: TestResult | None = None

    @property
    def success(self) -> bool:
        return self.status == RunStatus.SUCCESS

    @property
    def exit_code(self) -> int:
        if self.status == RunStatus.SUCCESS:
            return 0
        if self.status == RunStatus.CYCLE_LIMIT:
            return 2
        return 1


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        max_cycles: int | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.current_goal = goal
        self.settings = settings
        if self.settings is None:
            self.settings = getattr(client, "settings", None) or default_settings
        self.max_cycles = self.settings.max_cycles if max_cycles is None else max_cycles
        if self.max_cycles < 0:
            raise ValueError("max_cycles must be nonnegative")

        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self.last_result: RunResult | None = None
        self._last_worker_failed = False
        self._last_blocked_tasks = False

    async def run(self) -> RunResult:
        self.cycles = 0
        self.tasks = []
        self.last_test_result = None
        self.last_result = None
        self._last_worker_failed = False
        self._last_blocked_tasks = False

        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}", title="Furrow"))

        while self.max_cycles == 0 or self.cycles < self.max_cycles:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            try:
                test_result = await self._cycle()
            except Exception as exc:
                message = f"Run failed during cycle {self.cycles}: {exc}"
                console.print(f"[red]{message}[/red]")
                result = RunResult(
                    status=RunStatus.FAILURE,
                    cycles=self.cycles,
                    message=message,
                    test_result=self.last_test_result,
                )
                self.last_result = result
                return result

            if self._is_done():
                message = "Goal complete. Halting."
                console.print(f"[bold green]{message}[/bold green]")
                result = RunResult(
                    status=RunStatus.SUCCESS,
                    cycles=self.cycles,
                    message=message,
                    test_result=test_result,
                )
                self.last_result = result
                return result

            if test_result is None and self.tasks and not self._has_runnable_tasks():
                message = "No tasks were planned and no runnable tasks remain."
                console.print(f"[red]{message}[/red]")
                result = RunResult(
                    status=RunStatus.FAILURE,
                    cycles=self.cycles,
                    message=message,
                )
                self.last_result = result
                return result

            if self._last_worker_failed:
                message = "One or more worker tasks failed."
                result = RunResult(
                    status=RunStatus.FAILURE,
                    cycles=self.cycles,
                    message=message,
                    test_result=test_result,
                )
                self.last_result = result
                return result

            if self._last_blocked_tasks:
                message = "One or more tasks are blocked by their dependencies."
                result = RunResult(
                    status=RunStatus.FAILURE,
                    cycles=self.cycles,
                    message=message,
                    test_result=test_result,
                )
                self.last_result = result
                return result

            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                message = f"Cycle limit of {self.max_cycles} reached."
                console.print(f"[yellow]{message}[/yellow]")
                result = RunResult(
                    status=RunStatus.CYCLE_LIMIT,
                    cycles=self.cycles,
                    message=message,
                    test_result=test_result,
                )
                self.last_result = result
                return result

            if test_result is not None and not test_result.passed and not self._has_runnable_tasks():
                message = "Tests failed and no runnable tasks remain."
                console.print(f"[red]{message}[/red]")
                result = RunResult(
                    status=RunStatus.FAILURE,
                    cycles=self.cycles,
                    message=message,
                    test_result=test_result,
                )
                self.last_result = result
                return result

        message = f"Cycle limit of {self.max_cycles} reached."
        result = RunResult(status=RunStatus.CYCLE_LIMIT, cycles=self.cycles, message=message)
        self.last_result = result
        return result

    async def _cycle(self) -> TestResult | None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.current_goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._merge_plan(plan)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return None

        worker_failed, blocked = await self._execute_pending_tasks()
        self._last_worker_failed = worker_failed
        self._last_blocked_tasks = blocked

        try:
            tester = TesterAgent(client=self.client, settings=self.settings)
            test_result = await tester.run(self.current_goal, self.tasks)
        except Exception as exc:
            message = f"Tester failed: {exc}"
            console.print(f"[red]{message}[/red]")
            test_result = TestResult(passed=False, summary=str(exc), failures=[str(exc)])
            self.last_test_result = test_result
            raise RuntimeError(message) from exc

        self.last_test_result = test_result
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  [red]• {failure}[/red]")
            console.print("[yellow]Will attempt fix in the next cycle.[/yellow]")
            self.current_goal = (
                "Fix failing tests while preserving the original goal.\n\n"
                f"Original goal:\n{self.original_goal}\n\n"
                f"Test failures:\n" + "\n".join(test_result.failures)
            )
            self.goal = self.current_goal

        if worker_failed:
            raise RuntimeError("Worker task failures prevent a successful run")
        if blocked:
            raise RuntimeError("Blocked tasks prevent a successful run")
        return test_result

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
            if "status" in incoming.model_fields_set:
                if current.status == "completed" and incoming.status == "pending":
                    continue
                current.status = incoming.status
                current.result = incoming.result
            elif current.status in {"failed", "blocked"} and incoming.status == "pending":
                current.status = "pending"
                current.result = None

    async def _execute_pending_tasks(self) -> tuple[bool, bool]:
        semaphore = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))
        worker_failed = False
        blocked = False

        while True:
            ready: list[TaskModel] = []
            blocked_tasks: list[tuple[TaskModel, str]] = []
            by_id = {task.id: task for task in self.tasks}

            for task in self.tasks:
                if task.status != "pending":
                    continue

                dependency_tasks = [by_id.get(dependency) for dependency in task.dependencies]
                unavailable = [
                    dependency
                    for dependency, dependency_task in zip(task.dependencies, dependency_tasks)
                    if dependency_task is None
                    or dependency_task.status in {"failed", "blocked"}
                ]
                if unavailable:
                    reason = "missing or unavailable dependencies: " + ", ".join(unavailable)
                    blocked_tasks.append((task, reason))
                    continue
                if all(dependency_task.status == "completed" for dependency_task in dependency_tasks):
                    ready.append(task)

            for task, reason in blocked_tasks:
                task.status = "blocked"
                task.result = reason
                console.print(f"[red]Task {task.id} blocked: {reason}[/red]")
            blocked = blocked or bool(blocked_tasks)

            if not ready:
                break

            async def execute_task(task: TaskModel) -> None:
                nonlocal worker_failed
                async with semaphore:
                    task.status = "running"
                    console.print(f"[cyan]Task {task.id} running[/cyan]")
                    try:
                        worker = self._make_worker(task)
                        result = await worker.run()
                    except Exception as exc:
                        task.status = "failed"
                        task.result = str(exc)
                        worker_failed = True
                        console.print(f"[red]Task {task.id} failed: {exc}[/red]")
                    else:
                        if task.status in {"failed", "blocked"}:
                            worker_failed = True
                        task.status = "completed" if task.status not in {"failed", "blocked"} else task.status
                        task.result = result if isinstance(result, str) else str(result)
                        console.print(
                            f"[green]Task {task.id} completed[/green]"
                            if task.status == "completed"
                            else f"[red]Task {task.id} failed[/red]"
                        )

            await asyncio.gather(*(execute_task(task) for task in ready))
            self._print_task_states()

        if any(task.status == "pending" for task in self.tasks):
            blocked = True
        return worker_failed, blocked

    def _make_worker(self, task: TaskModel) -> WorkerAgent:
        try:
            return WorkerAgent(task=task, client=self.client, settings=self.settings)
        except TypeError:
            return WorkerAgent(task=task, client=self.client)

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

    def _is_done(self) -> bool:
        if not self.tasks:
            return True
        if any(task.status in {"pending", "running", "failed", "blocked"} for task in self.tasks):
            return False
        return self.last_test_result is not None and self.last_test_result.passed

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
