from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

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
    TaskStatus,
    TestResult,
    validate_task_graph,
)
from furrow.llm import LLMClient

console = Console()


class CycleLimitError(Exception):
    """Raised when the orchestrator exceeds the maximum number of cycles."""
    pass


class DeadlockError(Exception):
    """Raised when the orchestrator cannot make progress due to blocked tasks."""
    pass


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: Optional[LLMClient] = None,
        settings: Optional[Settings] = None,
        planner: Optional[PlannerAgent] = None,
        worker_factory: Optional[Callable[[TaskModel], Any]] = None,
        tester: Optional[TesterAgent] = None,
    ) -> None:
        self.original_goal = goal
        self.current_goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or Settings()
        self.planner = planner or PlannerAgent(client=self.client, settings=self.settings)
        self.tester = tester or TesterAgent(client=self.client, settings=self.settings)
        if worker_factory is None:
            self.worker_factory: Callable[[TaskModel], Any] = (
                lambda task: WorkerAgent(
                    task=task, client=self.client, settings=self.settings
                ).run()
            )
        else:
            self.worker_factory = worker_factory
        self.cycles = 0
        self.tasks: dict[str, TaskModel] = {}
        self.task_order: list[str] = []
        self.last_test_result: Optional[TestResult] = None
        self._repair_mode = False

    def _coerce_plan(self, value: Any) -> Plan:
        if isinstance(value, Plan):
            plan = value
        elif isinstance(value, dict):
            plan = Plan.model_validate(value)
        else:
            raise ValueError(f"Expected Plan or mapping, got {type(value).__name__}")
        plan.validate()
        return plan

    def _merge_plan(self, plan: Plan) -> None:
        if self._repair_mode and not plan.tasks:
            raise DeadlockError(
                "Repair mode active but planner returned no tasks to fix; "
                "cannot make progress on failing tests"
            )
        new_tasks: dict[str, TaskModel] = {}
        new_order: list[str] = []

        for task in plan.tasks:
            existing = self.tasks.get(task.id)
            preserve_completed = (
                existing is not None
                and existing.status == TaskStatus.COMPLETED
                and not self._repair_mode
            )
            if preserve_completed:
                new_tasks[task.id] = existing
            else:
                task = task.model_copy(deep=True)
                task.status = TaskStatus.PENDING
                task.error = None
                task.result = None
                new_tasks[task.id] = task

            if task.id not in new_order:
                new_order.append(task.id)

        for task_id in new_order:
            if task_id not in self.task_order:
                self.task_order.append(task_id)

        merged: dict[str, TaskModel] = {**self.tasks, **new_tasks}
        validate_task_graph(list(merged.values()))

        for task_id, task in new_tasks.items():
            self.tasks[task_id] = task

    def _get_tasks(self) -> list[TaskModel]:
        return [
            self.tasks[tid].model_copy(deep=True)
            for tid in self.task_order
            if tid in self.tasks
        ]

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        if self.last_test_result is None or not self.last_test_result.passed:
            return False
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks.values())

    def _ready_tasks(self) -> list[TaskModel]:
        completed_ids = {tid for tid, t in self.tasks.items() if t.status == TaskStatus.COMPLETED}
        ready = []
        for tid in self.task_order:
            task = self.tasks.get(tid)
            if not task or task.status != TaskStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in task.dependencies):
                ready.append(task)
        return ready

    def _mark_blocked(self) -> None:
        while True:
            blocked_ids = {
                tid
                for tid, t in self.tasks.items()
                if t.status in (TaskStatus.FAILED, TaskStatus.BLOCKED)
            }
            new_blocked = False
            for tid in self.task_order:
                task = self.tasks.get(tid)
                if not task or task.status in (
                    TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.BLOCKED
                ):
                    continue
                blocking = [dep for dep in task.dependencies if dep in blocked_ids]
                if blocking:
                    task.block(
                        f"Dependency blocked/failed: {', '.join(blocking)}"
                    )
                    new_blocked = True
            if not new_blocked:
                break

    async def _run_batch(self, tasks: list[TaskModel]) -> None:
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def run_one(task: TaskModel) -> None:
            async with semaphore:
                task.start()
                try:
                    result = await self.worker_factory(task)
                    result_str = str(result) if result is not None else ""
                    task.complete(result_str)
                except Exception as exc:
                    task.fail(str(exc))

        await asyncio.gather(*[run_one(t) for t in tasks])

    async def _execute_ready_tasks(self) -> None:
        ready = self._ready_tasks()
        if not ready:
            unfinished = any(
                t.status in (TaskStatus.PENDING, TaskStatus.FAILED)
                for t in self.tasks.values()
            )
            if unfinished:
                self._mark_blocked()
                ready = self._ready_tasks()
                if not ready:
                    raise DeadlockError("No ready tasks and unfinished work remains")
        await self._run_batch(ready)

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}",
                title="Furrow",
            )
        )

        while True:
            self.cycles += 1
            console.print(
                f"\n[bold cyan]\u2550\u2550\u2550 Cycle {self.cycles} "
                f"\u2550\u2550\u2550[/bold cyan]"
            )

            with Status("[bold yellow]Planning...", console=console):
                plan = await self.planner.plan(self.current_goal)

            plan = self._coerce_plan(plan)
            self._merge_plan(plan)
            console.print(
                Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
            )

            if not self.tasks and not self.last_test_result:
                raise DeadlockError("No tasks planned and no prior test result")

            await self._execute_ready_tasks()

            with Status("[bold yellow]Testing...", console=console):
                self.last_test_result = await self.tester.run(
                    self.original_goal, self._get_tasks()
                )

            if self.last_test_result.passed:
                self._repair_mode = False
                console.print(f"[green]Tests passed: {self.last_test_result.summary}[/green]")
            else:
                self._repair_mode = True
                console.print(f"[red]Tests failed: {self.last_test_result.summary}[/red]")
                for failure in self.last_test_result.failures:
                    console.print(f"  \u2022 {failure}")
                console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
                failures_text = "\n".join(self.last_test_result.failures)
                self.current_goal = (
                    f"Original goal: {self.original_goal}\n\n"
                    f"Fix failing tests:\n{failures_text}"
                )

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

            if self.cycles >= self.settings.max_cycles:
                raise CycleLimitError(
                    f"Reached max cycles ({self.settings.max_cycles}) "
                    f"without completing goal"
                )