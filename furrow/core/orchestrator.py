from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()
log = structlog.get_logger("furrow.orchestrator")

LogCallback = Callable[[str, dict[str, Any]], None]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_log: LogCallback | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.settings: Settings = self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._current_tasks: list[TaskModel] = []
        self._semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        self._on_log = on_log
        self._last_test_result: TestResult | None = None

    def _emit(self, event: str, level: str = "info", **data: Any) -> None:
        getattr(log, level, log.info)(event, **data)
        if self._on_log is not None:
            self._on_log(event, data)

    def _send(self, message: str) -> None:
        """Push an arbitrary message to any registered log callback."""
        if self._on_log is not None:
            self._on_log("message", {"message": message})

    async def run(self) -> None:
        self._emit("orchestrator_start", goal=self.goal, max_cycles=self.settings.max_cycles)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._emit("cycle_start", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._log_cycle_summary()
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self._emit("max_cycles_reached", level="warning", max_cycles=self.settings.max_cycles)
                console.print(
                    f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Halting.[/yellow]"
                )
                break
            if self._is_done():
                self._emit("goal_complete", cycle=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> Plan:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._emit("plan_created", cycle=self.cycles, num_tasks=len(plan.tasks))
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._send("No tasks planned. Goal may be complete.")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return plan

        self._current_tasks = plan.tasks
        ordered = self._topological_sort(self._current_tasks)
        self._emit("tasks_ordered", cycle=self.cycles, order=[t.id for t in ordered])

        await self._execute_tasks(ordered)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        await self._handle_test_result(test_result, plan.tasks)
        return plan

    async def _execute_tasks(self, tasks: list[TaskModel]) -> None:
        self._emit("executing_tasks", count=len(tasks), max_parallel=self.settings.max_parallel_tasks)
        completed_ids: set[str] = set()
        remaining = list(tasks)
        while remaining:
            ready = [t for t in remaining if all(d in completed_ids for d in t.dependencies)]
            if not ready:
                self._emit("dependency_cycle_detected", level="warning", remaining=[t.id for t in remaining])
                ready = remaining
            coros = [self._run_task(task) for task in ready]
            results = await asyncio.gather(*coros, return_exceptions=True)
            for task, result in zip(ready, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    self._emit("task_failed", level="error", task_id=task.id, error=str(result))
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result if isinstance(result, str) else str(result)
                    self._emit("task_completed", task_id=task.id)
                    console.print(f"[green]Task {task.id} completed[/green]")
            completed_ids.update(t.id for t in ready)
            remaining = [t for t in remaining if t.id not in completed_ids]

    async def _run_task(self, task: TaskModel) -> str:
        async with self._semaphore:
            self._emit("task_started", level="debug", task_id=task.id)
            return await WorkerAgent(task=task, client=self.client).run()

    def _topological_sort(self, tasks: list[TaskModel]) -> list[TaskModel]:
        by_id = {t.id: t for t in tasks}
        in_degree = {t.id: 0 for t in tasks}
        graph: dict[str, list[str]] = {t.id: [] for t in tasks}
        for task in tasks:
            for dep in task.dependencies:
                if dep in by_id:
                    graph[dep].append(task.id)
                    in_degree[task.id] += 1
        queue: deque[str] = deque(tid for tid, deg in in_degree.items() if deg == 0)
        order: list[str] = []
        while queue:
            tid = queue.popleft()
            order.append(tid)
            for nxt in graph[tid]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)
        ordered = [by_id[tid] for tid in order if tid in by_id]
        remaining = [t for t in tasks if t.id not in set(order)]
        ordered.extend(remaining)
        return ordered

    async def _handle_test_result(
        self, test_result: TestResult, tasks: list[TaskModel]
    ) -> None:
        self._last_test_result = test_result
        if test_result.passed:
            self._emit("tests_passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit("tests_failed", level="warning", failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _log_cycle_summary(self) -> None:
        table = Table(title=f"Cycle {self.cycles} Summary")
        table.add_column("Task ID", style="bold")
        table.add_column("Status")
        table.add_column("Result", overflow="ellipsis")
        for task in self._current_tasks:
            table.add_row(task.id, task.status, task.result or "")
        console.print(table)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            # All tasks completed, now check test result if available
            return not (self._last_test_result is not None and not self._last_test_result.passed)
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self._current_tasks
