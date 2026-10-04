from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()

EventSink = Callable[[str, dict[str, Any]], None]

# The loop is unbounded by default (max_cycles == 0), so the failure list fed
# back to the planner must not grow without limit.
_MAX_TRACKED_FAILURES = 25


def _jsonable(value: Any) -> Any:
    """Coerce a value into something ``json.dumps`` accepts."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


class Orchestrator:
    """Drives the plan -> execute -> test loop until the goal is met.

    The loop is intentionally unbounded; ``Settings.max_cycles`` (``0``) is the
    runaway guard for the advertised infinite loop. Task fan-out is capped by
    ``Settings.max_parallel_tasks``.

    Every message goes through :meth:`_emit`, so the ``rich`` output and the
    optional :data:`EventSink` stream can never drift apart. Emitted events:
    ``start``, ``cycle_start``, ``plan``, ``parallel_limit``, ``task_start``,
    ``task_complete``, ``task_failed``, ``test_result``, ``test_failure``,
    ``retry_planned``, ``cycle_end`` and ``halt``. Payloads are plain JSON types
    so callers can forward them verbatim (e.g. over a websocket).
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: EventSink | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.on_event = on_event
        self.cycles = 0
        self.plan: Plan | None = None
        self.last_test_result: TestResult | None = None
        self.failures: list[str] = []

    async def run(self) -> None:
        self._emit(
            "start",
            f"Goal: {self.goal}",
            renderable=Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"
            ),
            goal=self.goal,
        )
        while True:
            self.cycles += 1
            self._emit(
                "cycle_start",
                f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]",
                cycle=self.cycles,
            )
            await self._cycle()
            if self._is_done():
                self._emit(
                    "halt",
                    "[bold green]Goal complete. Halting.[/bold green]",
                    reason="goal_complete",
                    cycles=self.cycles,
                )
                return
            if self._budget_exhausted():
                budget = self._max_cycles()
                self._emit(
                    "halt",
                    f"[bold yellow]Cycle budget exhausted after {self.cycles} cycle(s) "
                    f"(FURROW_MAX_CYCLES={budget}). Stopping.[/bold yellow]",
                    reason="max_cycles",
                    cycles=self.cycles,
                    max_cycles=budget,
                )
                return

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self._planner_goal())
        self.plan = plan
        self._emit(
            "plan",
            f"Planned {len(plan.tasks)} task(s).",
            renderable=Panel(
                Pretty(plan.model_dump()), title="Plan", border_style="blue"
            ),
            rationale=plan.rationale,
            task_count=len(plan.tasks),
            tasks=[task.model_dump() for task in plan.tasks],
        )

        if not plan.tasks:
            self._emit(
                "cycle_end",
                "[yellow]No tasks planned. Goal may be complete.[/yellow]",
                cycle=self.cycles,
                task_count=0,
                passed=False,
                done=False,
            )
            return

        await self._execute_tasks(plan)
        await self._test(plan)
        self._emit(
            "cycle_end",
            f"Cycle {self.cycles} complete.",
            cycle=self.cycles,
            task_count=len(plan.tasks),
            passed=bool(self.last_test_result and self.last_test_result.passed),
            done=self._is_done(),
        )

    async def _execute_tasks(self, plan: Plan) -> None:
        limit = self._max_parallel_tasks()
        total = len(plan.tasks)
        if total > limit:
            self._emit(
                "parallel_limit",
                f"Throttling {total} tasks to {limit} concurrent workers.",
                total=total,
                limit=limit,
            )
        semaphore = asyncio.Semaphore(limit)

        async def run_task(task: TaskModel) -> str:
            async with semaphore:
                self._emit(
                    "task_start",
                    f"[cyan]→ {task.id}: {task.description}[/cyan]",
                    task_id=task.id,
                    description=task.description,
                )
                return await self._run_task(task)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *(run_task(task) for task in plan.tasks), return_exceptions=True
            )

        for task, result in zip(plan.tasks, results, strict=True):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                self._emit(
                    "task_failed",
                    f"[red]Task {task.id} failed: {result}[/red]",
                    task_id=task.id,
                    error=str(result),
                )
            else:
                task.status = "completed"
                task.result = result
                self._emit(
                    "task_complete",
                    f"[green]Task {task.id} completed[/green]",
                    task_id=task.id,
                    result=str(result),
                )

    async def _test(self, plan: Plan) -> None:
        with Status("[bold yellow]Testing...", console=console):
            test_result = await self._run_tests(plan)
        self.last_test_result = test_result
        if test_result.passed:
            message = f"[green]Tests passed: {test_result.summary}[/green]"
        else:
            message = f"[red]Tests failed: {test_result.summary}[/red]"
        self._emit(
            "test_result",
            message,
            passed=test_result.passed,
            summary=test_result.summary,
            failures=list(test_result.failures),
        )
        if test_result.passed:
            return
        for failure in test_result.failures:
            self._emit("test_failure", f"  • {failure}", failure=failure)
        self.failures = (self.failures + list(test_result.failures))[-_MAX_TRACKED_FAILURES:]
        self._emit(
            "retry_planned",
            "[yellow]Will attempt fix in next cycle.[/yellow]",
            pending_failures=len(self.failures),
        )

    async def _run_task(self, task: TaskModel) -> str:
        """Run one task. Seam for fakes; uses the real worker by default."""
        return await WorkerAgent(task=task, client=self.client).run()

    async def _run_tests(self, plan: Plan) -> TestResult:
        """Run the suite. Seam for fakes; uses the real tester by default."""
        return await TesterAgent(client=self.client).run(self.goal, plan.tasks)

    def _planner_goal(self) -> str:
        """Immutable goal plus the accumulated failures to resolve."""
        if not self.failures:
            return self.goal
        known = "\n".join(f"- {failure}" for failure in self.failures)
        return f"{self.goal}\n\nKnown failures to resolve:\n{known}"

    def _max_cycles(self) -> int:
        return max(0, int(self.client.settings.max_cycles or 0))

    def _budget_exhausted(self) -> bool:
        budget = self._max_cycles()
        return budget > 0 and self.cycles >= budget

    def _max_parallel_tasks(self) -> int:
        return max(1, int(self.client.settings.max_parallel_tasks or 1))

    def _is_done(self) -> bool:
        """True only for a non-empty plan with every task done and green tests."""
        tasks = self._get_tasks()
        if not tasks:
            return False
        if any(task.status == "failed" for task in tasks):
            return False
        if not all(task.status == "completed" for task in tasks):
            return False
        test_result = self.last_test_result
        return test_result is not None and test_result.passed

    def _get_tasks(self) -> list[Any]:
        if self.plan is None:
            return []
        return list(self.plan.tasks)

    def _emit(
        self,
        event: str,
        message: str = "",
        /,
        *,
        renderable: Any = None,
        **data: Any,
    ) -> None:
        """Print ``message`` (or ``renderable``) and forward ``event`` to the sink."""
        console.print(message if renderable is None else renderable)
        if self.on_event is None:
            return
        payload: dict[str, Any] = {"event": event, "message": message}
        payload.update({key: _jsonable(value) for key, value in data.items()})
        self.on_event(event, payload)
