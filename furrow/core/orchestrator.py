from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

console = Console()

MAX_FAILURE_CONTEXT_LINES = 20
MAX_EMPTY_PLAN_CYCLES = 3

# An event sink may be a plain function or a coroutine function (the web layer owns a
# websocket, so its sink is async). Both are supported.
EventCallback = Callable[[str], None] | Callable[[str], Awaitable[None]]


class Orchestrator:
    """Runs the plan -> execute -> test loop until the goal is satisfied."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        *,
        on_event: EventCallback | None = None,
    ) -> None:
        # self.goal is immutable for the lifetime of the orchestrator: failing test
        # output is kept in _failure_context so the original goal is never lost.
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self._tasks: list[TaskModel] = []
        self._results: dict[str, str] = {}
        self._seen: set[str] = set()
        self._failure_context: list[str] = []
        self._empty_plan_cycles = 0
        self._stop_requested = False
        self._last_test_passed: bool | None = None
        self._on_event = on_event

    # ------------------------------------------------------------------ loop

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            if self._stop_requested:
                await self._emit("stop requested, halting")
                console.print("[yellow]Stop requested. Halting.[/yellow]")
                break

            self.cycles += 1
            cycle = self.cycles
            await self._emit(f"cycle {cycle} starting")
            console.print(f"\n[bold cyan]═══ Cycle {cycle} ═══[/bold cyan]")

            await self._cycle()

            if self._is_done():
                await self._emit("goal complete, halting")
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

            max_cycles = int(getattr(self.client.settings, "max_cycles", 0) or 0)
            if max_cycles > 0 and cycle >= max_cycles:
                await self._emit(f"max cycles ({max_cycles}) reached, halting")
                console.print(f"[yellow]Reached max cycles ({max_cycles}). Halting.[/yellow]")
                break

            if self._empty_plan_cycles >= MAX_EMPTY_PLAN_CYCLES:
                await self._emit(
                    f"no new tasks for {self._empty_plan_cycles} cycles, halting"
                )
                console.print(
                    f"[yellow]No new tasks planned for {self._empty_plan_cycles} cycles. "
                    "Halting.[/yellow]"
                )
                break

    async def _cycle(self) -> None:
        previous_failures = "\n".join(self._failure_context)
        self._failure_context = []

        with Status("[bold yellow]Planning...", console=console):
            try:
                plan = await self._plan(previous_failures)
            except Exception as exc:  # noqa: BLE001 - a bad plan must not kill the loop
                self._empty_plan_cycles += 1
                message = f"{type(exc).__name__}: {exc}"
                await self._emit(
                    f"planner failed: {message} "
                    f"({self._empty_plan_cycles}/{MAX_EMPTY_PLAN_CYCLES})"
                )
                console.print(
                    f"[red]Planning failed ({self._empty_plan_cycles}/"
                    f"{MAX_EMPTY_PLAN_CYCLES}):[/red] {message}"
                )
                return

        self.plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit(f"plan produced with {len(plan.tasks)} task(s)")

        new_tasks = self._record_tasks(plan.tasks)
        if not new_tasks:
            self._empty_plan_cycles += 1
            await self._emit(
                f"no new tasks planned ({self._empty_plan_cycles}/{MAX_EMPTY_PLAN_CYCLES})"
            )
            console.print("[yellow]No new tasks planned.[/yellow]")
            return
        self._empty_plan_cycles = 0

        await self._execute(new_tasks, asyncio.Semaphore(self._parallel_limit()))

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, new_tasks)

        self._last_test_passed = bool(test_result.passed)
        if test_result.passed:
            await self._emit(f"tests passed: {test_result.summary}")
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._failure_context = self._failure_lines(test_result.summary, test_result.failures)
            await self._emit(f"tests failed: {test_result.summary}")
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")

    # ------------------------------------------------------------- internals

    async def _plan(self, context: str) -> Plan:
        # Called through a loose signature: the planner may or may not accept context.
        plan: Callable[..., Awaitable[Plan]] = self.planner.plan
        return await plan(self.goal, context=context)

    def _record_tasks(self, planned: list[TaskModel]) -> list[TaskModel]:
        """Append not-yet-seen tasks to the persistent list, skipping duplicates."""
        fresh: list[TaskModel] = []
        for task in planned:
            signature = self._signature(task)
            if signature in self._seen:
                continue
            self._seen.add(signature)
            task.status = "pending"
            self._tasks.append(task)
            fresh.append(task)
        return fresh

    async def _execute(self, tasks: list[TaskModel], semaphore: asyncio.Semaphore) -> None:
        async def run_task(task: TaskModel) -> str:
            async with semaphore:
                await self._emit(f"task {task.id} started: {task.description}")
                console.print(f"[bold]{task.id}[/bold] {task.description}")
                return await WorkerAgent(task=task, client=self.client).run()

        with Status(f"[bold yellow]Executing {len(tasks)} task(s)...", console=console):
            coroutines = [run_task(task) for task in tasks]
            results: list[str | BaseException] = await asyncio.gather(
                *coroutines, return_exceptions=True
            )

        for task, result in zip(tasks, results):
            if isinstance(result, BaseException):
                message = str(result)
                task.status = "failed"
                task.result = message
                self._results[task.id] = message
                console.print(f"[red]Task {task.id} failed: {message}[/red]")
                await self._emit(f"task {task.id} failed: {message}")
            else:
                task.status = "completed"
                task.result = result
                self._results[task.id] = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._emit(f"task {task.id} completed")

    def _parallel_limit(self) -> int:
        limit = int(getattr(self.client.settings, "max_parallel_tasks", 0) or 0)
        return max(1, limit)

    @staticmethod
    def _signature(task: TaskModel) -> str:
        description = " ".join(task.description.split()).lower()
        return description or f"id:{task.id}"

    @staticmethod
    def _failure_lines(summary: str, failures: list[str]) -> list[str]:
        lines = [f"Fix failing tests: {summary}"]
        lines.extend(f"- {failure}" for failure in failures)
        return lines[-MAX_FAILURE_CONTEXT_LINES:]

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        return all(task.status == "completed" for task in tasks) and self._last_test_passed is True

    def _get_tasks(self) -> list[TaskModel]:
        return list(self._tasks)

    async def _emit(self, message: str) -> None:
        """Forward an event to the sink, awaiting it when the sink is a coroutine.

        A raising or disconnected consumer must never kill the loop.
        """
        callback = self._on_event
        if callback is None:
            return
        with contextlib.suppress(Exception):
            result = callback(message)
            if inspect.isawaitable(result):
                await result

    def stop(self) -> None:
        """Request the loop to halt before the next cycle."""
        self._stop_requested = True