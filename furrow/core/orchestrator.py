from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TaskStatus, TestResult
from furrow.llm import LLMClient

logger = logging.getLogger(__name__)
console = Console()

# A consumer may register a sync callback or an async one; both are invoked as
# handler(event_name, payload).
EventHandler = Callable[[str, dict[str, Any]], Awaitable[None] | None]

# Consecutive cycles stuck in the *same* state (identical test failures, identical failing
# tasks, planner errors, or crashed cycles) tolerated before the loop declares itself stuck and
# halts instead of burning API credits forever.
STALL_LIMIT = 3
DEFAULT_MAX_PARALLEL = 5
# Stall fingerprints are compared as text, so each fragment is whitespace-collapsed, lowercased
# and truncated: cosmetic drift between cycles must not hide a stuck loop, while genuinely
# different work must still produce a different fingerprint.
SIGNATURE_TEXT_LIMIT = 120

__all__ = ["EventHandler", "LoopResult", "Orchestrator"]


def _fingerprint(text: str) -> str:
    """Normalize ``text`` for stall comparison: whitespace-collapsed, lowercased, truncated."""
    return " ".join(text.split()).lower()[:SIGNATURE_TEXT_LIMIT]


@dataclass(slots=True)
class LoopResult:
    goal: str
    cycles: int
    stopped_reason: str  # "goal-complete" | "no-tasks" | "max-cycles" | "stalled" | "cancelled"
    passed: bool
    tasks_completed: int = 0
    tasks_failed: int = 0
    failures: list[str] = field(default_factory=list)


class Orchestrator:
    """Drives the plan -> execute -> test loop for a single goal.

    ``goal`` is the standing objective and is never rewritten. ``focus`` holds
    the failure-driven instruction for the current cycle, so every plan prompt
    contains both the original goal and what failed last cycle.

    Every cycle outcome either makes progress or counts toward a stall: a cycle
    that repeats the previous stuck state (same test failures, same failing tasks)
    :data:`STALL_LIMIT` times in a row halts the loop instead of spinning forever.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: EventHandler | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.on_event = on_event
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.focus: str | None = None
        self._last_test_result: TestResult | None = None
        self._planner_empty = False
        self._cycle_failed = 0
        self._failure_signature: tuple[str, ...] | None = None
        self._failure_repeat = 0
        self._planner_failures = 0
        self._cycle_errors = 0

    async def run(self) -> LoopResult:
        """Run the loop until it stops and always return a LoopResult.

        Cancellation contract: ``asyncio.CancelledError`` is caught, in-flight
        workers are cancelled, and a ``LoopResult`` with
        ``stopped_reason="cancelled"`` is returned rather than re-raised, so
        asyncio-based callers (CLI, web layer) always receive a result.
        ``KeyboardInterrupt`` is deliberately allowed to propagate so the CLI's
        existing ``except KeyboardInterrupt`` path keeps working.
        """
        await self._emit("run.start", {
            "message": f"Starting Furrow loop for goal: {self.goal}",
            "goal": self.goal,
        })
        banner = Panel.fit(
            f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"
        )
        console.print(banner)
        reason = "goal-complete"
        stall_message: str | None = None
        try:
            while True:
                max_cycles = self._max_cycles()
                if max_cycles > 0 and self.cycles >= max_cycles:
                    reason = "max-cycles"
                    break
                self.cycles += 1
                await self._emit("cycle.start", {
                    "message": f"Starting cycle {self.cycles}",
                    "cycle": self.cycles,
                })
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                try:
                    await self._cycle()
                    self._cycle_errors = 0
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._cycle_errors += 1
                    message = f"Cycle {self.cycles} crashed: {exc}"
                    logger.exception(message)
                    console.print(f"[red]{message}[/red]")
                    await self._emit("cycle.done", {"message": message, "cycle": self.cycles})
                if self._is_done():
                    reason = "no-tasks" if self._planner_empty else "goal-complete"
                    break
                stall_message = self._stall_reason()
                if stall_message is not None:
                    reason = "stalled"
                    break
        except asyncio.CancelledError:
            self._cancel_pending_tasks()
            reason = "cancelled"
            stall_message = None

        result = self._build_result(reason, stall_message)
        if reason == "goal-complete":
            final_message = "Goal complete. Halting."
        elif reason == "no-tasks":
            final_message = "Planner produced no tasks. Nothing left to do."
        elif reason == "max-cycles":
            final_message = f"Cycle limit reached ({self._max_cycles()}). Halting."
        elif reason == "stalled":
            final_message = stall_message or "Loop stalled. Halting."
        else:
            final_message = "Run cancelled."
        color = "green" if reason == "goal-complete" else "yellow"
        console.print(f"[bold {color}]{final_message}[/bold {color}]")
        await self._emit("run.done", {
            "message": final_message,
            "cycles": self.cycles,
            "reason": reason,
        })
        return result

    async def _cycle(self) -> None:
        self._cycle_failed = 0
        self._planner_empty = False
        with Status("[bold yellow]Planning...", console=console):
            try:
                plan = await self.planner.plan(self._planning_goal())
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001  # deliberate: non-fatal cycle
                self._planner_failures += 1
                message = f"Planner failed in cycle {self.cycles}: {exc}"
                logger.warning(message)
                console.print(f"[red]{message}[/red]")
                await self._emit("cycle.done", {"message": message, "cycle": self.cycles})
                return
        self._planner_failures = 0
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit("plan", {
            "message": f"Cycle {self.cycles} planned {len(plan.tasks)} task(s)",
            "cycle": self.cycles,
            "tasks": [task.id for task in plan.tasks],
            "rationale": plan.rationale,
        })
        self.tasks.extend(plan.tasks)
        if not plan.tasks:
            self._planner_empty = True
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._emit("cycle.done", {
                "message": "Planner produced no tasks; nothing left to do",
                "cycle": self.cycles,
            })
            return
        await self._execute_tasks(plan.tasks)
        with Status("[bold yellow]Testing...", console=console):
            try:
                test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001  # deliberate: non-fatal tester
                test_result = TestResult(passed=False, summary=str(exc), failures=[str(exc)])
        self._last_test_result = test_result
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
        failure_notes = list(test_result.failures)
        if self._cycle_failed:
            failure_notes.extend(
                f"Task {task.id} failed: {task.result}"
                for task in plan.tasks
                if task.status == TaskStatus.FAILED
            )
        if failure_notes:
            self.focus = "Fix the following failures:\n" + "\n".join(failure_notes)
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
        else:
            self.focus = None
        outcome = "passed" if test_result.passed else "failed"
        await self._emit("test.result", {
            "message": f"Tests {outcome}: {test_result.summary}",
            "passed": test_result.passed,
            "summary": test_result.summary,
            "failures": list(test_result.failures),
        })
        self._record_cycle_signature(test_result, plan.tasks)
        await self._emit("cycle.done", {
            "message": f"Cycle {self.cycles} finished",
            "cycle": self.cycles,
        })

    async def _execute_tasks(self, plan_tasks: list[TaskModel]) -> None:
        """Run plan tasks honoring dependencies, bounded by max_parallel_tasks.

        A task starts only once every task it names in ``dependencies`` has
        completed successfully; a task whose dependency failed or was skipped is
        marked ``skipped``. Independent tasks run concurrently, capped by a
        semaphore. Unknown ids or dependency cycles would deadlock the scheduler,
        so the remainder is failed explicitly instead of hanging.
        """
        semaphore = asyncio.Semaphore(self._max_parallel())
        by_id: dict[str, TaskModel] = {task.id: task for task in plan_tasks}
        pending = set(by_id)
        completed: set[str] = set()
        failed: set[str] = set()
        skipped: set[str] = set()
        running: dict[str, asyncio.Task[None]] = {}

        while pending or running:
            for task_id in sorted(pending):
                task = by_id[task_id]
                if any(dep in failed or dep in skipped for dep in task.dependencies):
                    pending.discard(task_id)
                    skipped.add(task_id)
                    task.status = TaskStatus.SKIPPED
                    task.result = "Dependency failed or was skipped"
                    console.print(f"[yellow]Task {task_id} skipped[/yellow]")
                    await self._emit("task.skipped", {
                        "message": f"Task {task_id} skipped: dependency failed or was skipped",
                        "task_id": task_id,
                        "reason": "Dependency failed or was skipped",
                    })
            for task_id in sorted(pending):
                task = by_id[task_id]
                if all(dep in completed for dep in task.dependencies):
                    pending.discard(task_id)
                    running[task_id] = asyncio.create_task(self._run_worker(task, semaphore))
            if not running:
                for task_id in sorted(pending):
                    task = by_id[task_id]
                    pending.discard(task_id)
                    failed.add(task_id)
                    task.status = TaskStatus.FAILED
                    task.result = "Unresolvable dependencies (unknown id or dependency cycle)"
                    self._cycle_failed += 1
                    console.print(f"[red]Task {task_id} failed: {task.result}[/red]")
                    await self._emit("task.failed", {
                        "message": f"Task {task_id} failed: {task.result}",
                        "task_id": task_id,
                        "error": task.result,
                    })
                break
            try:
                done, _ = await asyncio.wait(
                    set(running.values()), return_when=asyncio.FIRST_COMPLETED
                )
            except asyncio.CancelledError:
                for worker in running.values():
                    worker.cancel()
                await asyncio.gather(*running.values(), return_exceptions=True)
                raise
            for worker in done:
                task_id = next(tid for tid, w in running.items() if w is worker)
                del running[task_id]
                task = by_id[task_id]
                if worker.cancelled():
                    task.status = TaskStatus.FAILED
                    task.result = "Worker cancelled"
                    failed.add(task_id)
                    self._cycle_failed += 1
                elif task.status == TaskStatus.COMPLETED:
                    completed.add(task_id)
                else:
                    failed.add(task_id)
                    self._cycle_failed += 1

    async def _run_worker(self, task: TaskModel, semaphore: asyncio.Semaphore) -> None:
        async with semaphore:
            await self._emit("task.start", {
                "message": f"Starting task {task.id}: {task.description}",
                "task_id": task.id,
                "description": task.description,
            })
            try:
                result = await WorkerAgent(task=task, client=self.client).run()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001  # deliberate: non-fatal worker
                task.status = TaskStatus.FAILED
                task.result = str(exc)
                console.print(f"[red]Task {task.id} failed: {exc}[/red]")
                await self._emit("task.failed", {
                    "message": f"Task {task.id} failed: {exc}",
                    "task_id": task.id,
                    "error": str(exc),
                })
            else:
                task.status = TaskStatus.COMPLETED
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._emit("task.done", {
                    "message": f"Task {task.id} completed",
                    "task_id": task.id,
                    "files": list(task.files),
                    "status": task.status,
                })

    def _planning_goal(self) -> str:
        if self.focus is None:
            return self.goal
        return f"{self.goal}\n\nCurrent focus:\n{self.focus}"

    def _is_done(self) -> bool:
        if self._planner_empty:
            return True
        if self._last_test_result is None:
            return False
        if self._cycle_failed > 0:
            return False
        return self._last_test_result.passed

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks

    def _stall_reason(self) -> str | None:
        if self._failure_repeat >= STALL_LIMIT:
            return self._describe_stall()
        if self._planner_failures >= STALL_LIMIT:
            return f"Loop stalled: planner failed {STALL_LIMIT} consecutive cycles"
        if self._cycle_errors >= STALL_LIMIT:
            return f"Loop stalled: {STALL_LIMIT} consecutive cycles crashed unexpectedly"
        return None

    def _describe_stall(self) -> str:
        """Name the repeated stuck state so the reason is actionable, not just "stalled"."""
        # ``_failure_repeat`` only advances on a non-empty signature, and every fragment
        # ``_cycle_signature`` builds carries exactly one of these two prefixes, so the two
        # branches below cover every stall that can be reported here.
        signature = self._failure_signature or ()
        stuck_tasks = [part for part in signature if part.startswith("failed task ")]
        if not stuck_tasks:
            stuck_tests = [part for part in signature if part.startswith("test failure: ")]
            return (
                f"Loop stalled: identical test failures for {STALL_LIMIT} consecutive "
                f"cycles ({'; '.join(stuck_tests)})"
            )
        message = (
            f"Loop stalled: the same tasks keep failing for {STALL_LIMIT} consecutive "
            f"cycles ({'; '.join(stuck_tasks)})"
        )
        stuck_tests = [part for part in signature if part.startswith("test failure: ")]
        if stuck_tests:
            message += f"; identical test failures repeated too ({'; '.join(stuck_tests)})"
        return message

    def _cycle_signature(
        self, test_result: TestResult, cycle_tasks: Sequence[TaskModel]
    ) -> tuple[str, ...]:
        """Fingerprint the stuck state of one cycle as one sorted tuple of text fragments.

        Test failures and failed tasks are folded into the *same* fingerprint, so a single
        code path (:meth:`_stall_reason`) decides that the loop is stuck no matter which half of
        the cycle is failing.

        A failing task is keyed on its description plus its error text and never on
        ``TaskModel.id``: the planner mints fresh ids every cycle, so an id-keyed fingerprint
        would differ every cycle and the stall would never trip.
        """
        fragments = [
            f"test failure: {_fingerprint(failure)}" for failure in test_result.failures
        ]
        fragments.extend(
            f"failed task {_fingerprint(task.description)}: "
            f"{_fingerprint(task.result or '')}"
            for task in cycle_tasks
            if task.status == TaskStatus.FAILED
        )
        return tuple(sorted(fragments))

    def _record_cycle_signature(
        self, test_result: TestResult, cycle_tasks: Sequence[TaskModel]
    ) -> None:
        signature = self._cycle_signature(test_result, cycle_tasks)
        if not signature:
            # Nothing failed this cycle: real progress, so any earlier stuck state is cleared.
            self._failure_signature = None
            self._failure_repeat = 0
            return
        if signature == self._failure_signature:
            self._failure_repeat += 1
        else:
            self._failure_signature = signature
            self._failure_repeat = 1

    def _cancel_pending_tasks(self) -> None:
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                task.result = "Run cancelled before the task completed"

    def _build_result(self, reason: str, stall_message: str | None = None) -> LoopResult:
        failures: list[str] = []
        if self._last_test_result is not None:
            failures.extend(self._last_test_result.failures)
        failures.extend(
            f"Task {task.id} failed: {task.result}"
            for task in self.tasks
            if task.status == TaskStatus.FAILED
        )
        if stall_message is not None:
            failures.append(stall_message)
        return LoopResult(
            goal=self.goal,
            cycles=self.cycles,
            stopped_reason=reason,
            # Only a genuine goal-complete stop means the objective was achieved;
            # "no-tasks" after unresolved failures, stalls, limits and
            # cancellations all report passed=False.
            passed=(
                reason == "goal-complete"
                and self._last_test_result is not None
                and self._last_test_result.passed
                and self._cycle_failed == 0
            ),
            tasks_completed=sum(1 for task in self.tasks if task.status == TaskStatus.COMPLETED),
            tasks_failed=sum(1 for task in self.tasks if task.status == TaskStatus.FAILED),
            failures=failures,
        )

    def _max_cycles(self) -> int:
        limit = getattr(self.client.settings, "max_cycles", 0)
        return limit if isinstance(limit, int) and limit > 0 else 0

    def _max_parallel(self) -> int:
        limit = getattr(self.client.settings, "max_parallel_tasks", DEFAULT_MAX_PARALLEL)
        if not isinstance(limit, int) or limit < 1:
            return DEFAULT_MAX_PARALLEL
        return limit

    async def _emit(self, name: str, payload: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            outcome = self.on_event(name, payload)
            if inspect.isawaitable(outcome):
                await outcome
        except Exception:
            # A broken or disconnected consumer must never take the loop down.
            logger.debug("Event handler raised for %s", name, exc_info=True)
