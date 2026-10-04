"""Cycle orchestration for the Furrow autonomous development loop.

The orchestrator drives a repeating plan/execute/test loop. Every cycle it asks
the planner for the next slice of the *immutable* goal, executes those tasks
concurrently while honouring the dependencies they declare, runs the tester,
and folds any failures back into the next cycle's context. The loop halts when
the goal is genuinely met, when ``settings.max_cycles`` is exhausted, when a
caller asks it to :meth:`Orchestrator.stop`, or when it detects that it is no
longer making progress.

Nothing here talks to a provider directly: all model traffic goes through
:class:`furrow.llm.LLMClient`, which is also responsible for its own cleanup.
"""

from __future__ import annotations

import asyncio
import contextlib
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TaskStatus, TestResult, get_settings
from furrow.llm import LLMClient

__all__ = ["EventCallback", "Orchestrator"]

#: Async sink for structured progress events, used by the web UI.
EventCallback = Callable[[dict[str, Any]], Awaitable[None]]

#: A task id is abandoned after this many failures inside the whole run.
MAX_TASK_ATTEMPTS = 3
#: Consecutive cycles that neither complete a task nor pass their tests.
MAX_CONSECUTIVE_FAILED_CYCLES = 5

_default_console = Console()
#: Backwards-compatible module-level console.
console = _default_console


def _describe(exc: BaseException) -> str:
    """Render an exception as a short, always-non-empty string."""
    text = str(exc).strip()
    name = type(exc).__name__
    return f"{name}: {text}" if text else name


class Orchestrator:
    """Run plan/execute/test cycles until the goal is met or a limit is hit."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        console: Console | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self._owns_client = client is None
        self.console: Console = console if console is not None else _default_console
        self.on_event = on_event
        client_settings = getattr(self.client, "settings", None)
        self.settings: Settings = (
            client_settings if isinstance(client_settings, Settings) else get_settings()
        )
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.current_focus: str | None = None
        self.last_test_result: TestResult | None = None
        self._task_failures: dict[str, tuple[int, str]] = {}
        self._exhausted: dict[str, str] = {}
        self._consecutive_failed_cycles = 0
        self._terminal_reason: str | None = None
        self._stop_requested = False

    # -- public API --------------------------------------------------------

    async def run(self) -> None:
        """Drive cycles until the loop is done. Cleans up a self-owned client."""
        self.console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        await self._emit("loop_started", goal=self.goal)
        reason = "goal complete"
        try:
            while not self._stop_requested:
                self.cycles += 1
                self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                await self._emit("cycle_started", goal=self.goal, focus=self.current_focus)
                try:
                    await self._cycle()
                except (KeyboardInterrupt, asyncio.CancelledError):
                    raise
                except Exception as exc:  # noqa: BLE001 - a bad cycle must not kill the loop
                    self._consecutive_failed_cycles += 1
                    self.console.print(
                        f"[red]Cycle {self.cycles} failed: {_describe(exc)}[/red]"
                    )
                    await self._emit(
                        "cycle_failed",
                        error=_describe(exc),
                        traceback=traceback.format_exc(),
                    )
                if self._stop_requested:
                    reason = "stop requested by caller"
                    break
                if self._is_done():
                    reason = self._stop_reason()
                    if reason == "goal complete":
                        self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                    else:
                        self.console.print(f"[bold yellow]Stopping: {reason}.[/bold yellow]")
                    break
        except BaseException as exc:
            reason = f"aborted: {_describe(exc)}"
            raise
        finally:
            await self._emit(
                "loop_stopped",
                reason=reason,
                tasks=[task.model_dump(mode="json") for task in self.tasks],
            )
            if self._owns_client:
                with contextlib.suppress(Exception):
                    await self.client.aclose()

    async def stop(self) -> None:
        """Ask the loop to exit at the next cycle boundary."""
        self._stop_requested = True
        self.console.print("[yellow]Stop requested; halting after this cycle.[/yellow]")
        await self._emit("stop_requested")

    # -- cycle -------------------------------------------------------------

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console):
            plan = await self.planner.plan(self._build_retry_goal())
        self.tasks = list(plan.tasks)
        self._terminal_reason = None
        if plan.rationale:
            self.console.print(
                Panel(plan.rationale, title="Rationale", border_style="blue")
            )
        self.console.print(
            Panel(Pretty(plan.model_dump(mode="json")), title="Plan", border_style="blue")
        )
        await self._emit(
            "plan_created",
            rationale=plan.rationale,
            tasks=[task.model_dump(mode="json") for task in self.tasks],
        )
        if not self.tasks:
            self._terminal_reason = "planner returned no tasks"
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return
        if self._halt_if_exhausted():
            return

        with Status("[bold yellow]Executing tasks...", console=self.console):
            await self._execute_tasks(self.tasks)
        if self._halt_if_exhausted():
            return

        with Status("[bold yellow]Testing...", console=self.console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)
        self.last_test_result = test_result
        await self._emit(
            "tests_completed",
            passed=test_result.passed,
            summary=test_result.summary,
            failures=list(test_result.failures),
        )
        progressed = any(t.status == TaskStatus.COMPLETED for t in self.tasks)
        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self.current_focus = None
            self._consecutive_failed_cycles = 0
            return

        self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
        for failure in test_result.failures:
            self.console.print(f"  • {failure}")
        self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
        self.current_focus = self._build_test_focus(test_result)
        if progressed:
            self._consecutive_failed_cycles = 0
        else:
            self._consecutive_failed_cycles += 1
            if self._consecutive_failed_cycles >= MAX_CONSECUTIVE_FAILED_CYCLES:
                self._terminal_reason = (
                    f"{self._consecutive_failed_cycles} consecutive cycles made no "
                    "progress and left failing tests"
                )
                self.console.print(
                    f"[bold red]{self._terminal_reason}; halting.[/bold red]"
                )

    # -- scheduling --------------------------------------------------------

    async def _execute_tasks(self, tasks: list[TaskModel]) -> None:
        by_id = self._index_tasks(tasks)
        deps = {task.id: self._resolve_dependencies(task, by_id) for task in tasks}
        semaphore = asyncio.Semaphore(max(1, int(self.settings.max_parallel_tasks)))
        remaining = [t for t in tasks if t.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED)]
        in_flight: dict[asyncio.Task[None], TaskModel] = {}
        fatal: BaseException | None = None
        try:
            while remaining or in_flight:
                launched = False
                for task in list(remaining):
                    if all(by_id[dep].status == TaskStatus.COMPLETED for dep in deps[task.id]):
                        remaining.remove(task)
                        in_flight[asyncio.create_task(self._run_task(task, semaphore))] = task
                        launched = True
                if not in_flight:
                    if launched:
                        continue
                    await self._fail_blocked(remaining, deps, by_id)
                    break
                done, _ = await asyncio.wait(
                    set(in_flight), return_when=asyncio.FIRST_COMPLETED
                )
                for finished in done:
                    in_flight.pop(finished, None)
                    error = self._future_error(finished)
                    if error is not None and not isinstance(error, Exception):
                        fatal = error
                        break
                if fatal is not None:
                    break
        finally:
            await self._drain(in_flight)
        if fatal is not None:
            raise fatal

    def _index_tasks(self, tasks: list[TaskModel]) -> dict[str, TaskModel]:
        by_id: dict[str, TaskModel] = {}
        for task in tasks:
            if task.id in by_id:
                self.console.print(
                    f"[yellow]Duplicate task id '{task.id}' in plan; "
                    "dependencies resolve to the first occurrence.[/yellow]"
                )
                continue
            by_id[task.id] = task
        return by_id

    def _resolve_dependencies(
        self, task: TaskModel, by_id: dict[str, TaskModel]
    ) -> list[str]:
        """Keep only dependency ids that exist in this plan; warn about the rest."""
        resolved: list[str] = []
        for dep in task.dependencies:
            if dep == task.id:
                self.console.print(
                    f"[yellow]Task {task.id} declares itself as a dependency; ignored.[/yellow]"
                )
                continue
            if dep not in by_id:
                self.console.print(
                    f"[yellow]Task {task.id} depends on unknown task '{dep}'; ignored.[/yellow]"
                )
                continue
            if dep not in resolved:
                resolved.append(dep)
        return resolved

    async def _run_task(self, task: TaskModel, semaphore: asyncio.Semaphore) -> None:
        try:
            async with semaphore:
                task.status = TaskStatus.RUNNING
                await self._emit("task_started", task_id=task.id, description=task.description)
                result = await WorkerAgent(task=task, client=self.client).run()
        except (KeyboardInterrupt, asyncio.CancelledError) as exc:
            # Record the failure, then let the interruption unwind the loop.
            task.status = TaskStatus.FAILED
            task.result = _describe(exc)
            self._note_failure(task, task.result)
            self.console.print(f"[red]Task {task.id} interrupted: {task.result}[/red]")
            await self._emit("task_failed", task_id=task.id, error=task.result, fatal=True)
            raise
        except Exception as exc:  # noqa: BLE001 - one bad task must not sink the cycle
            task.status = TaskStatus.FAILED
            task.result = _describe(exc)
            self._note_failure(task, task.result)
            self.console.print(f"[red]Task {task.id} failed: {task.result}[/red]")
            await self._emit("task_failed", task_id=task.id, error=task.result)
        else:
            task.status = TaskStatus.COMPLETED
            task.result = result if isinstance(result, str) else str(result)
            self.console.print(f"[green]Task {task.id} completed[/green]")
            await self._emit("task_completed", task_id=task.id, result=task.result)

    async def _fail_blocked(
        self,
        remaining: list[TaskModel],
        deps: dict[str, list[str]],
        by_id: dict[str, TaskModel],
    ) -> None:
        """No runnable task remains: report the deadlock instead of spinning."""
        for task in remaining:
            blockers = [dep for dep in deps[task.id] if by_id[dep].status != TaskStatus.COMPLETED]
            failed = [dep for dep in blockers if by_id[dep].status == TaskStatus.FAILED]
            if failed:
                reason = f"blocked by failed task(s): {', '.join(failed)}"
            else:
                reason = f"blocked by uncompleted dependency: {', '.join(blockers)}"
            task.status = TaskStatus.FAILED
            task.result = reason
            self.console.print(f"[red]Task {task.id} not run: {reason}[/red]")
            await self._emit("task_failed", task_id=task.id, error=reason, blocked=True)
        remaining.clear()

    async def _drain(self, in_flight: dict[asyncio.Task[None], TaskModel]) -> None:
        """Cancel and await leftover workers so nothing leaks out of a cycle."""
        if not in_flight:
            return
        for future, task in in_flight.items():
            future.cancel()
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.FAILED
                task.result = "cancelled before it started"
        with contextlib.suppress(Exception):
            await asyncio.gather(*in_flight, return_exceptions=True)
        in_flight.clear()

    @staticmethod
    def _future_error(future: asyncio.Task[None]) -> BaseException | None:
        if future.cancelled():
            return asyncio.CancelledError()
        return future.exception()

    def _note_failure(self, task: TaskModel, error: str) -> None:
        """Track consecutive failures per task and retire the ones that never work."""
        count, _ = self._task_failures.get(task.id, (0, ""))
        count += 1
        self._task_failures[task.id] = (count, error)
        if count >= MAX_TASK_ATTEMPTS and task.id not in self._exhausted:
            self._exhausted[task.id] = error
            self.console.print(
                f"[bold red]Task {task.id} has failed {count} time(s); "
                "not retrying it again.[/bold red]"
            )

    def _halt_if_exhausted(self) -> bool:
        """Stop instead of paying for another planner call that cannot help."""
        if not self.tasks or not all(task.id in self._exhausted for task in self.tasks):
            return False
        self._terminal_reason = "every planned task has failed repeatedly"
        self.console.print(
            "[bold red]Every planned task has failed repeatedly; halting.[/bold red]"
        )
        return True

    # -- goal / context ----------------------------------------------------

    def _build_retry_goal(self) -> str:
        """Original goal, plus the failure context the next plan should absorb.

        ``self.goal`` is never rewritten, so the user's objective survives every
        cycle even when the tester returns an empty ``failures`` list.
        """
        sections = [f"Goal: {self.goal}"]
        if self.current_focus:
            sections.append(self.current_focus)
        if self._task_failures:
            lines = [
                f"- task {task_id} (failed {count} time(s)): {error}"
                for task_id, (count, error) in self._task_failures.items()
            ]
            sections.append("Task failures from previous cycles to fix:\n" + "\n".join(lines))
        if self._exhausted:
            sections.append(
                "These task areas already failed repeatedly, take a different "
                f"approach instead of repeating them: {', '.join(sorted(self._exhausted))}"
            )
        return "\n\n".join(sections)

    @staticmethod
    def _build_test_focus(test_result: TestResult) -> str:
        """Turn a failed test run into focus text, falling back to its summary."""
        detail = "\n".join(f"* {failure}" for failure in test_result.failures)
        if not detail:
            detail = test_result.summary.strip() or "the tester gave no detail"
        return (
            "The previous cycle's tests failed. Fix these failures before continuing:\n"
            f"{detail}"
        )

    # -- state -------------------------------------------------------------

    def _is_done(self) -> bool:
        """True when the loop must halt: a limit was hit or the cycle succeeded."""
        if self._stop_requested:
            return True
        if self._cycle_cap_reached():
            return True
        if self._terminal_reason is not None:
            return True
        if any(task.status == TaskStatus.FAILED for task in self._get_tasks()):
            return False
        return self.last_test_result is not None and self.last_test_result.passed

    def _cycle_cap_reached(self) -> bool:
        max_cycles = self.settings.max_cycles
        return max_cycles > 0 and self.cycles >= max_cycles

    def _stop_reason(self) -> str:
        if self._stop_requested:
            return "stop requested by caller"
        if self._cycle_cap_reached():
            return f"cycle cap of {self.settings.max_cycles} reached"
        if self._terminal_reason is not None:
            return self._terminal_reason
        return "goal complete"

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks

    # -- events ------------------------------------------------------------

    async def _emit(self, event: str, **data: Any) -> None:
        if self.on_event is None:
            return
        payload: dict[str, Any] = {"event": event, "cycle": self.cycles, **data}
        try:
            await self.on_event(payload)
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception as exc:  # noqa: BLE001 - a bad UI sink cannot break the loop
            self.console.print(
                f"[yellow]on_event callback for '{event}' failed: {_describe(exc)}[/yellow]"
            )
