from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TaskStatus, settings
from furrow.core.events import Event
from furrow.llm import LLMClient

console = Console()

REASON_NO_TASKS = "no tasks planned — goal satisfied"
REASON_COMPLETE = "goal complete"
REASON_MAX_CYCLES = "max cycles reached"
REASON_NO_PROGRESS = "no progress"


@dataclass
class _CycleResult:
    """Outcome of a single orchestration cycle."""

    planned: int = 0
    completed: int = 0
    failed: int = 0
    passed: bool = False
    summary: str = ""
    failures: list[str] = field(default_factory=list)
    error: str = ""
    satisfied: bool = False

    @property
    def made_progress(self) -> bool:
        return self.completed > 0


class Orchestrator:
    """Drives the plan -> execute -> test loop until the goal is met.

    The orchestrator keeps real state across cycles (``self.tasks`` holds every
    task ever planned, ``self.history`` one entry per cycle) so the loop can
    iterate instead of halting after the first pass.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[Event], None] | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client: LLMClient = client if client is not None else LLMClient()
        self.on_event = on_event
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.history: list[dict[str, Any]] = []
        self._consecutive_failures = 0

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}",
                title="Furrow",
            )
        )
        reason = REASON_COMPLETE
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit("cycle_start", {"cycle": self.cycles})
            result = await self._cycle()
            self._emit("cycle_end", {"cycle": self.cycles})

            if result.made_progress:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1

            stop_reason = self._stop_reason(result)
            if stop_reason is not None:
                reason = stop_reason
                break

        self._finish(reason)

    async def _cycle(self) -> _CycleResult:
        cycle = self.cycles
        try:
            with Status("[bold yellow]Planning...", console=console) as status:
                plan = await self.planner.plan(self.goal)
                status.update(f"[bold yellow]Plan received: {len(plan.tasks)} task(s)")
        except Exception as exc:
            message = f"planning failed: {exc}"
            console.print(f"[red]{message}[/red]")
            self._record_history(cycle, [], False, message, 0, 0)
            return _CycleResult(summary=message, error=message)

        planned = list(plan.tasks)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit(
            "plan",
            {
                "cycle": cycle,
                "tasks": [
                    {"id": t.id, "description": t.description, "files": list(t.files)}
                    for t in planned
                ],
                "rationale": plan.rationale,
            },
        )

        if not planned:
            console.print("[yellow]No tasks planned — the goal appears to be satisfied.[/yellow]")
            self._record_history(cycle, [], False, "No tasks planned.", 0, 0, satisfied=True)
            return _CycleResult(summary="No tasks planned.", satisfied=True)

        tasks = self._merge_tasks(planned)
        completed, failed = await self._execute(tasks)

        try:
            with Status("[bold yellow]Testing...", console=console) as status:
                test_result = await TesterAgent(client=self.client).run(self.goal, tasks)
                status.update("[bold yellow]Test run finished")
        except Exception as exc:
            message = f"testing failed: {exc}"
            console.print(f"[red]{message}[/red]")
            self._emit(
                "test_result",
                {"cycle": cycle, "passed": False, "summary": message, "failures": [message]},
            )
            self._record_history(cycle, [t.id for t in tasks], False, message, completed, failed)
            return _CycleResult(
                planned=len(tasks),
                completed=completed,
                failed=failed,
                summary=message,
                error=message,
            )

        passed = bool(test_result.passed)
        failures = list(test_result.failures)
        summary = test_result.summary
        self._emit(
            "test_result",
            {"cycle": cycle, "passed": passed, "summary": summary, "failures": failures},
        )
        if passed:
            console.print(f"[green]Tests passed: {summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {summary}[/red]")
            for failure in failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = self._narrowed_goal(failures)

        self._record_history(
            cycle,
            [t.id for t in tasks],
            passed,
            summary,
            completed,
            failed,
            failures=failures,
        )
        return _CycleResult(
            planned=len(tasks),
            completed=completed,
            failed=failed,
            passed=passed,
            summary=summary,
            failures=failures,
        )

    async def _execute(self, tasks: list[TaskModel]) -> tuple[int, int]:
        """Run the given tasks, at most ``settings.max_parallel_tasks`` at a time."""
        cap = max(1, self._settings().max_parallel_tasks)
        completed = 0
        failed = 0
        with Status(
            f"[bold yellow]Executing {len(tasks)} task(s) (max {cap} in parallel)...",
            console=console,
        ) as status:
            for offset in range(0, len(tasks), cap):
                chunk = tasks[offset : offset + cap]
                remaining = len(tasks) - offset
                status.update(
                    f"[bold yellow]Executing {remaining} task(s) remaining "
                    f"(max {cap} in parallel)..."
                )
                results = await asyncio.gather(
                    *[self._run_task(task) for task in chunk],
                    return_exceptions=True,
                )
                for task, result in zip(chunk, results, strict=True):
                    if isinstance(result, BaseException):
                        error = str(result)
                        task.status = TaskStatus.FAILED
                        task.result = error
                        failed += 1
                        console.print(f"[red]Task {task.id} failed: {error}[/red]")
                        self._emit(
                            "task_failed",
                            {"cycle": self.cycles, "task_id": task.id, "error": error},
                        )
                    else:
                        text = str(result)
                        task.status = TaskStatus.COMPLETED
                        task.result = text
                        completed += 1
                        console.print(f"[green]Task {task.id} completed[/green]")
                        self._emit(
                            "task_done",
                            {
                                "cycle": self.cycles,
                                "task_id": task.id,
                                "result": text,
                                "files_changed": list(task.files_changed),
                            },
                        )
        return completed, failed

    async def _run_task(self, task: TaskModel) -> str:
        task.status = TaskStatus.IN_PROGRESS
        self._emit(
            "task_start",
            {
                "cycle": self.cycles,
                "task_id": task.id,
                "description": task.description,
            },
        )
        worker = WorkerAgent(task=task, client=self.client)
        result = await worker.run()
        task.files_changed = list(worker.files_changed)
        return result

    def _merge_tasks(self, planned: list[TaskModel]) -> list[TaskModel]:
        """Fold planned tasks into :attr:`tasks`, updating known ids in place."""
        known = {task.id: task for task in self.tasks}
        merged: list[TaskModel] = []
        for candidate in planned:
            existing = known.get(candidate.id)
            if existing is None:
                existing = candidate
                self.tasks.append(existing)
                known[existing.id] = existing
            else:
                existing.description = candidate.description
                existing.files = list(candidate.files)
                existing.dependencies = list(candidate.dependencies)
                existing.status = TaskStatus.PENDING
                existing.result = None
            merged.append(existing)
        return merged

    def _narrowed_goal(self, failures: list[str]) -> str:
        details = "\n".join(f"- {failure}" for failure in failures) or "- (no details reported)"
        return (
            f"Original goal: {self.original_goal}\n\n"
            "The previous cycle left failing tests. Fix them without losing sight of the "
            "original goal.\n\n"
            f"Failing tests:\n{details}\n\n"
            f"Goal: {self.original_goal}"
        )

    def _stop_reason(self, result: _CycleResult) -> str | None:
        if not result.error and result.planned == 0:
            return REASON_NO_TASKS
        if result.passed and result.failed == 0 and result.completed == result.planned:
            return REASON_COMPLETE
        cfg = self._settings()
        if cfg.max_cycles > 0 and self.cycles >= cfg.max_cycles:
            return REASON_MAX_CYCLES
        if (
            cfg.max_consecutive_failures > 0
            and self._consecutive_failures >= cfg.max_consecutive_failures
        ):
            return REASON_NO_PROGRESS
        return None

    def _finish(self, reason: str) -> None:
        completed = sum(1 for task in self.tasks if task.status == TaskStatus.COMPLETED)
        failed = sum(1 for task in self.tasks if task.status == TaskStatus.FAILED)
        self._emit("done", {"cycles": self.cycles, "reason": reason})
        success = reason in (REASON_COMPLETE, REASON_NO_TASKS)
        state = "complete" if success else "incomplete"
        colour = "green" if success else "yellow"
        console.print(
            Panel(
                f"[bold]Cycles run:[/bold] {self.cycles}\n"
                f"[bold]Tasks completed:[/bold] {completed}\n"
                f"[bold]Tasks failed:[/bold] {failed}\n"
                f"[bold]Goal:[/bold] {self.original_goal}\n"
                f"[bold]Final status:[/bold] {state}\n"
                f"[bold]Stop reason:[/bold] {reason}",
                title="Summary",
                border_style=colour,
            )
        )
        if success:
            console.print("[bold green]Goal complete. Halting.[/bold green]")
        else:
            console.print(f"[bold yellow]Stopped: {reason}[/bold yellow]")

    def _record_history(
        self,
        cycle: int,
        task_ids: list[str],
        passed: bool,
        summary: str,
        completed: int,
        failed: int,
        failures: list[str] | None = None,
        satisfied: bool = False,
    ) -> None:
        self.history.append(
            {
                "cycle": cycle,
                "tasks": list(task_ids),
                "passed": passed,
                "summary": summary,
                "completed": completed,
                "failed": failed,
                "failures": list(failures or []),
                "satisfied": satisfied,
            }
        )

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        """Emit a progress event; a broken listener must never kill the run."""
        if self.on_event is None:
            return
        with contextlib.suppress(Exception):
            self.on_event(Event(kind=kind, data=dict(data)))

    def _settings(self) -> Settings:
        candidate = getattr(self.client, "settings", None)
        if isinstance(candidate, Settings):
            return candidate
        return settings
