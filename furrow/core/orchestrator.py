from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
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
    TaskStatus,
    TestResult,
    settings as default_settings,
)
from furrow.llm import LLMClient

console = Console()


class _ClientAdapter:
    def __init__(self, client: Any, settings: Settings) -> None:
        self._client = client
        self.settings = settings

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
    ) -> str:
        result = _call_with_supported_kwargs(
            self._client.complete,
            prompt,
            system=system,
            model=model,
        )
        if inspect.isawaitable(result):
            return await result
        return result

    async def read_file(self, path: str | Path) -> str:
        result = _call_with_supported_kwargs(self._client.read_file, path)
        if inspect.isawaitable(result):
            return await result
        return result

    async def write_file(self, path: str | Path, content: str) -> None:
        result = _call_with_supported_kwargs(
            self._client.write_file,
            path,
            content=content,
        )
        if inspect.isawaitable(result):
            await result

    def list_files(self, directory: str | Path) -> list[str]:
        return _call_with_supported_kwargs(self._client.list_files, directory)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _call_with_supported_kwargs(call: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(call)
    except (TypeError, ValueError):
        return call(*args, **kwargs)

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return call(*args, **kwargs)

    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return call(
        *args,
        **{name: value for name, value in kwargs.items() if name in accepted},
    )


def _factory_accepts_client(factory: Any) -> bool:
    if not callable(factory):
        return False
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return True
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return True
    return "client" in signature.parameters


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        planner: Any | None = None,
        worker_factory: Any | None = None,
        tester: Any | None = None,
        *,
        settings_obj: Settings | None = None,
        worker: Any | None = None,
        max_cycles: int | None = None,
        max_parallel_tasks: int | None = None,
    ) -> None:
        if settings is not None and settings_obj is not None:
            raise ValueError("pass only one of settings or settings_obj")

        self.original_goal = goal
        self.goal = goal
        client_settings = getattr(client, "settings", None)
        self.settings = (
            settings
            or settings_obj
            or client_settings
            or default_settings
        )
        self.client = (
            client if client is not None else LLMClient(settings=self.settings)
        )
        self._agent_client = (
            _ClientAdapter(self.client, self.settings)
            if client is not None
            else self.client
        )
        self.planner = (
            planner
            if planner is not None
            else PlannerAgent(client=self._agent_client)
        )
        self.tester = (
            tester
            if tester is not None
            else TesterAgent(client=self._agent_client)
        )
        if worker_factory is not None:
            self.worker_factory = worker_factory
        elif worker is not None:
            self.worker_factory = worker
        else:
            self.worker_factory = WorkerAgent

        self.max_cycles = (
            max_cycles
            if max_cycles is not None
            else self.settings.max_cycles
        )
        self.max_parallel_tasks = (
            max_parallel_tasks
            if max_parallel_tasks is not None
            else self.settings.max_parallel_tasks
        )
        if self.max_cycles < 0:
            raise ValueError("max_cycles must be >= 0")
        if self.max_parallel_tasks < 1:
            raise ValueError("max_parallel_tasks must be >= 1")

        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self.current_plan: Plan | None = None
        self.last_test_result: TestResult | None = None

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

            tasks = self._get_tasks()
            if any(
                self._status(task) == TaskStatus.FAILED for task in tasks
            ):
                console.print("[red]Task failure detected. Halting.[/red]")
                break

            if not tasks:
                console.print("[yellow]No tasks planned. Halting.[/yellow]")
                break

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

        if (
            self.max_cycles > 0
            and self.cycles >= self.max_cycles
            and not self._is_done()
        ):
            console.print(
                f"[yellow]Maximum cycles ({self.max_cycles}) reached. Halting.[/yellow]"
            )

    async def _cycle(self) -> None:
        plan_value = await self._run_planner()
        plan = self._coerce_plan(plan_value)
        self._merge_plan(plan)
        self.current_plan = plan
        console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        ready_tasks = self._ready_tasks()
        if ready_tasks:
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                await self._execute_ready_tasks(ready_tasks)

        self._fail_blocked_tasks()
        self._fail_stalled_tasks()

        with Status("[bold yellow]Testing...", console=console):
            test_result = await self._run_tester()
        self.last_test_result = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")

    async def _run_planner(self) -> Any:
        planner = getattr(self.planner, "plan", self.planner)
        result = planner(self.original_goal)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _run_tester(self) -> TestResult:
        tester = getattr(self.tester, "run", self.tester)
        result = tester(self.original_goal, self._get_tasks())
        if inspect.isawaitable(result):
            result = await result
        return self._coerce_test_result(result)

    def _coerce_plan(self, value: Any) -> Plan:
        if isinstance(value, Plan):
            return value
        if isinstance(value, dict):
            return Plan.model_validate(value)
        raise ValueError(f"expected Plan or mapping, got {type(value).__name__}")

    def _coerce_test_result(self, value: Any) -> TestResult:
        if isinstance(value, TestResult):
            return value
        if isinstance(value, dict):
            return TestResult.model_validate(value)
        raise ValueError(f"expected TestResult or mapping, got {type(value).__name__}")

    def _merge_plan(self, plan: Plan) -> None:
        merged = list(self._tasks)
        existing_by_id = {task.id: task for task in merged}

        for incoming in plan.tasks:
            existing = existing_by_id.get(incoming.id)
            if existing is None:
                task = incoming.model_copy(deep=True)
                task.status = TaskStatus.PENDING
                task.result = None
                merged.append(task)
                continue

            existing.description = incoming.description
            existing.files = list(incoming.files)
            existing.dependencies = list(incoming.dependencies)

        self._validate_task_graph(merged)
        self._tasks = merged

    def _find_dependency_cycle(self, tasks: list[TaskModel]) -> list[str] | None:
        dependencies = {task.id: list(task.dependencies) for task in tasks}
        visiting: set[str] = set()
        visited: set[str] = set()
        stack: list[str] = []

        def visit(task_id: str) -> list[str] | None:
            if task_id in visiting:
                return stack[stack.index(task_id) :] + [task_id]
            if task_id in visited:
                return None

            visiting.add(task_id)
            stack.append(task_id)
            for dependency in dependencies[task_id]:
                cycle = visit(dependency)
                if cycle is not None:
                    return cycle
            stack.pop()
            visiting.remove(task_id)
            visited.add(task_id)
            return None

        for task_id in dependencies:
            cycle = visit(task_id)
            if cycle is not None:
                return cycle
        return None

    def _validate_task_graph(self, tasks: list[TaskModel]) -> None:
        task_ids = [task.id for task in tasks]
        seen_ids: set[str] = set()
        duplicates: list[str] = []
        for task_id in task_ids:
            if task_id in seen_ids:
                duplicates.append(task_id)
            seen_ids.add(task_id)
        if duplicates:
            raise ValueError(f"duplicate task id: {duplicates[0]!r}")

        known_ids = set(task_ids)
        for task in tasks:
            dependencies = list(task.dependencies)
            if len(dependencies) != len(set(dependencies)):
                raise ValueError(f"task {task.id!r} has duplicate dependencies")
            for dependency in dependencies:
                if dependency not in known_ids:
                    raise ValueError(
                        f"task {task.id!r} has unknown dependency {dependency!r}"
                    )
                if dependency == task.id:
                    raise ValueError(f"task {task.id!r} depends on itself")

        cycle = self._find_dependency_cycle(tasks)
        if cycle is not None:
            raise ValueError(
                f"dependency cycle detected: {' -> '.join(cycle)}"
            )

    def _ready_tasks(self) -> list[TaskModel]:
        tasks_by_id = {task.id: task for task in self._tasks}
        ready: list[TaskModel] = []
        for task in self._tasks:
            if self._status(task) != TaskStatus.PENDING:
                continue
            if all(
                dependency in tasks_by_id
                and self._status(tasks_by_id[dependency]) == TaskStatus.COMPLETED
                for dependency in task.dependencies
            ):
                ready.append(task)
        return ready

    async def _execute_ready_tasks(self, ready_tasks: list[TaskModel]) -> None:
        semaphore = asyncio.Semaphore(self.max_parallel_tasks)

        async def run_one(task: TaskModel) -> None:
            async with semaphore:
                task.status = TaskStatus.RUNNING
                try:
                    result = await self._run_worker(task)
                except Exception as exc:
                    task.status = TaskStatus.FAILED
                    task.result = str(exc)
                    console.print(f"[red]Task {task.id} failed: {exc}[/red]")
                else:
                    task.status = TaskStatus.COMPLETED
                    task.result = "" if result is None else str(result)
                    console.print(f"[green]Task {task.id} completed[/green]")

        await asyncio.gather(
            *(run_one(task) for task in ready_tasks),
            return_exceptions=True,
        )

    async def _run_worker(self, task: TaskModel) -> Any:
        worker = self._build_worker(task)
        runner = getattr(worker, "run", None)
        if callable(runner):
            result = runner()
        else:
            result = worker
        if inspect.isawaitable(result):
            return await result
        return result

    def _build_worker(self, task: TaskModel) -> Any:
        if not callable(self.worker_factory):
            return self.worker_factory
        if _factory_accepts_client(self.worker_factory):
            return self.worker_factory(task=task, client=self._agent_client)
        return self.worker_factory(task)

    def _fail_blocked_tasks(self) -> None:
        tasks_by_id = {task.id: task for task in self._tasks}
        for task in self._tasks:
            if self._status(task) != TaskStatus.PENDING:
                continue

            failed_dependencies = [
                dependency
                for dependency in task.dependencies
                if dependency in tasks_by_id
                and self._status(tasks_by_id[dependency]) == TaskStatus.FAILED
            ]
            missing_dependencies = [
                dependency
                for dependency in task.dependencies
                if dependency not in tasks_by_id
            ]
            if not failed_dependencies and not missing_dependencies:
                continue

            reasons = []
            if failed_dependencies:
                reasons.append(
                    "dependency failed: " + ", ".join(failed_dependencies)
                )
            if missing_dependencies:
                reasons.append(
                    "dependency missing: " + ", ".join(missing_dependencies)
                )
            task.status = TaskStatus.FAILED
            task.result = "; ".join(reasons)
            console.print(f"[red]Task {task.id} failed: {task.result}[/red]")

    def _fail_stalled_tasks(self) -> None:
        pending_tasks = [
            task
            for task in self._tasks
            if self._status(task) == TaskStatus.PENDING
        ]
        if not pending_tasks or self._ready_tasks():
            return

        reason = "task stalled: no ready pending tasks remain"
        for task in pending_tasks:
            task.status = TaskStatus.FAILED
            task.result = reason
            console.print(f"[red]Task {task.id} failed: {reason}[/red]")

    def _status(self, task: TaskModel) -> TaskStatus:
        status = task.status
        if isinstance(status, TaskStatus):
            return status
        try:
            return TaskStatus(status)
        except ValueError:
            return TaskStatus.PENDING

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        return all(
            self._status(task) in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for task in tasks
        )

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks

    @property
    def tasks(self) -> list[TaskModel]:
        return self._get_tasks()

    @property
    def current_tasks(self) -> list[TaskModel]:
        return self._get_tasks()

    @property
    def plan(self) -> Plan | None:
        return self.current_plan
