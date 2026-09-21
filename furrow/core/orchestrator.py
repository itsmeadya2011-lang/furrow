from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

console = Console()

WorkerFactory = Callable[[TaskModel], Awaitable[str]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        planner: PlannerAgent | None = None,
        tester: TesterAgent | None = None,
        worker_factory: WorkerFactory | None = None,
    ) -> None:
        client_settings = getattr(client, "settings", None)
        self.settings = (
            settings
            if settings is not None
            else client_settings or default_settings
        )
        self.goal = goal
        self.original_goal = goal
        self.client = client if client is not None else LLMClient(settings=self.settings)
        self.planner = planner or PlannerAgent(client=self.client)
        self.tester = tester or TesterAgent(client=self.client)
        self.worker_factory = worker_factory
        self.cycles = 0
        self.max_cycles = max(0, self.settings.max_cycles)
        self.max_parallel_tasks = max(1, self.settings.max_parallel_tasks)
        self.tasks: list[TaskModel] = []
        self.current_tasks = self.tasks
        self.current_plan: Plan | None = None
        self.plan = self.current_plan
        self.latest_test_result: TestResult | None = None
        self.test_result = self.latest_test_result
        self.last_test_result = self.latest_test_result

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while self.max_cycles == 0 or self.cycles < self.max_cycles:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

        if self.max_cycles > 0 and self.cycles >= self.max_cycles and not self._is_done():
            console.print(
                f"[yellow]Maximum cycles ({self.max_cycles}) reached. Halting.[/yellow]"
            )

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)

        planned_tasks = list(plan.tasks)
        self._merge_plan(plan)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not planned_tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        results = await self._execute_tasks(planned_tasks)
        persistent_tasks = self._tasks_for_plan(planned_tasks)
        for task, result in zip(persistent_tasks, results):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await self.tester.run(self.goal, self._get_tasks())

        self._set_test_result(test_result)
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, planned_tasks: list[TaskModel]) -> list[object]:
        tasks = [
            task
            for task in self._tasks_for_plan(planned_tasks)
            if task.status != "completed"
        ]
        if not tasks:
            return []

        semaphore = asyncio.Semaphore(self.max_parallel_tasks)

        async def run_task(task: TaskModel) -> object:
            async with semaphore:
                if self.worker_factory is not None:
                    return await self.worker_factory(task)
                return await WorkerAgent(task=task, client=self.client).run()

        return await asyncio.gather(
            *(run_task(task) for task in tasks),
            return_exceptions=True,
        )

    def _merge_plan(self, plan: Plan) -> None:
        self.current_plan = plan
        self.plan = plan
        for planned_task in plan.tasks:
            existing = next(
                (task for task in self.tasks if task.id == planned_task.id),
                None,
            )
            if existing is None:
                planned_task.status = "pending"
                planned_task.result = None
                self.tasks.append(planned_task)
                continue

            existing.description = planned_task.description
            existing.files = list(planned_task.files)
            existing.dependencies = list(planned_task.dependencies)
            if existing.status == "failed" or (
                existing.status == "completed"
                and self.latest_test_result is not None
                and not self.latest_test_result.passed
            ):
                existing.status = "pending"
                existing.result = None

        if plan.tasks:
            plan.tasks = list(self.tasks)

    def _tasks_for_plan(self, planned_tasks: list[TaskModel]) -> list[TaskModel]:
        tasks_by_id = {task.id: task for task in self.tasks}
        tasks: list[TaskModel] = []
        seen: set[str] = set()
        for planned_task in planned_tasks:
            if planned_task.id in seen:
                continue
            task = tasks_by_id.get(planned_task.id)
            if task is not None:
                tasks.append(task)
                seen.add(planned_task.id)
        return tasks

    def _set_test_result(self, result: TestResult) -> None:
        self.latest_test_result = result
        self.test_result = result
        self.last_test_result = result

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        if any(task.status != "completed" for task in tasks):
            return False
        return self.latest_test_result is None or self.latest_test_result.passed

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
