from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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


@dataclass
class CycleRecord:
    """Immutable record of one completed cycle of the development loop."""

    cycle: int
    plan: Plan
    test_result: TestResult
    tasks: list[TaskModel]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        settings: Settings | None = None,
    ) -> None:
        # ``goal`` is the immutable original goal; ``directive`` is what the next
        # cycle plans against (the goal, or a test-fix instruction).
        self.goal = goal
        self.directive = goal
        self.on_event = on_event
        if settings is not None:
            self.settings = settings
        else:
            self.settings = getattr(client, "settings", None) or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.history: list[CycleRecord] = []
        self._running = False

    # ------------------------------------------------------------------ events

    def _emit(self, type: str, message: str, data: dict[str, Any] | None = None) -> None:
        """Emit a JSON-serializable progress event to ``on_event``, if provided."""
        if self.on_event is None:
            return
        event: dict[str, Any] = {
            "type": type,
            "cycle": self.cycles,
            "message": message,
            "data": dict(data or {}),
        }
        # Deliberately broad: a broken observer must never take the development
        # loop down, so every failure mode of on_event is swallowed here.
        with contextlib.suppress(Exception):
            self.on_event(event)

    # ------------------------------------------------------------------- state

    def _limit_reached(self) -> bool:
        max_cycles = self.settings.max_cycles
        return max_cycles > 0 and self.cycles >= max_cycles

    def _last_cycle_complete(self) -> bool:
        if not self.history:
            return False
        tasks = self.history[-1].tasks
        if not tasks:
            # An empty plan is not evidence of completion.
            return False
        return all(task.status == "completed" for task in tasks)

    def _is_done(self) -> bool:
        """True when the cycle budget is exhausted or the last cycle fully succeeded."""
        if self._limit_reached():
            return True
        return self._last_cycle_complete()

    def status(self) -> dict[str, Any]:
        """JSON-serializable snapshot of the loop for programmatic consumers."""
        record = self.history[-1] if self.history else None
        last_test: dict[str, Any] | None = None
        if record is not None:
            last_test = {
                "passed": record.test_result.passed,
                "summary": record.test_result.summary,
                "failures": list(record.test_result.failures),
            }
        tasks = [
            {
                "id": task.id,
                "description": task.description,
                "status": task.status,
                "result": task.result,
            }
            for task in (record.tasks if record is not None else [])
        ]
        return {
            "goal": self.goal,
            "directive": self.directive,
            "cycles": self.cycles,
            "running": self._running,
            "done": self._is_done(),
            "tasks": tasks,
            "last_test": last_test,
        }

    # --------------------------------------------------------------------- run

    async def run(self) -> None:
        self._running = True
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        try:
            while True:
                self.cycles += 1
                self._emit(
                    "cycle_start",
                    f"Cycle {self.cycles} started",
                    {"goal": self.goal, "directive": self.directive},
                )
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

                record = await self._cycle()

                if record is None:
                    # The planner produced nothing: halt instead of spinning forever.
                    self._halt("Planner produced no tasks. Halting to avoid an unbounded loop.")
                    break

                if self._limit_reached():
                    if self._last_cycle_complete():
                        self._finish()
                    else:
                        self._halt(
                            f"Reached the max_cycles limit ({self.settings.max_cycles}). Halting."
                        )
                    break

                if self._is_done():
                    self._finish()
                    break
        except asyncio.CancelledError:
            self._emit("run_halt", "Run cancelled.", {})
            raise
        except Exception as exc:
            self._emit("run_halt", f"Run aborted after error: {exc}", {"error": str(exc)})
            raise
        finally:
            self._running = False

    async def _cycle(self) -> CycleRecord | None:
        """Run one plan → execute → test cycle. Returns None if the plan was empty."""
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.directive)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit(
            "plan",
            f"Planned {len(plan.tasks)} task(s)",
            {"rationale": plan.rationale, "tasks": [task.model_dump() for task in plan.tasks]},
        )

        if not plan.tasks:
            console.print("[yellow]No tasks planned. The planner is out of work; halting.[/yellow]")
            self._emit("cycle_end", "Cycle ended with no tasks planned.", {"tasks": 0})
            return None

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await self._execute(plan.tasks)

        # ``results`` is built by _execute with one slot per task, so the lengths
        # always match; strict=True keeps a future refactor from silently
        # dropping a task's outcome.
        for task, result in zip(plan.tasks, results, strict=True):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            # Always test against the original goal, never a mutated one.
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self._emit(
            "test_result",
            test_result.summary,
            {
                "passed": test_result.passed,
                "summary": test_result.summary,
                "failures": list(test_result.failures),
            },
        )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self.directive = self.goal
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Only the directive changes; self.goal stays the original goal.
            self.directive = "Fix failing tests:\n" + "\n".join(test_result.failures)

        record = CycleRecord(
            cycle=self.cycles,
            plan=plan,
            test_result=test_result,
            tasks=list(plan.tasks),
        )
        self.history.append(record)
        self._emit(
            "cycle_end",
            f"Cycle {self.cycles} complete",
            {
                "passed": test_result.passed,
                "completed": sum(1 for t in plan.tasks if t.status == "completed"),
                "failed": sum(1 for t in plan.tasks if t.status == "failed"),
            },
        )
        return record

    async def _execute(self, tasks: list[TaskModel]) -> list[Any]:
        """Run the workers, keeping at most ``max_parallel_tasks`` in flight."""
        limit = self.settings.max_parallel_tasks
        if limit is None or limit < 1:
            limit = 1
        limit = min(limit, len(tasks))

        results: list[Any] = [None] * len(tasks)
        queue: asyncio.Queue[tuple[int, TaskModel]] = asyncio.Queue()
        for index, task in enumerate(tasks):
            queue.put_nowait((index, task))

        async def runner() -> None:
            while True:
                try:
                    index, task = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                outcome: Any = None
                status = "completed"
                try:
                    self._emit(
                        "task_start",
                        f"Task {task.id} started",
                        {"id": task.id, "description": task.description},
                    )
                    outcome = await WorkerAgent(task=task, client=self.client).run()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # surfaced per task, never aborts the cycle
                    outcome = exc
                    status = "failed"
                results[index] = outcome
                self._emit(
                    "task_end",
                    f"Task {task.id} {status}",
                    {
                        "id": task.id,
                        "status": status,
                        "result": None if isinstance(outcome, BaseException) else outcome,
                    },
                )

        if limit:
            await asyncio.gather(*(runner() for _ in range(limit)))
        return results

    # ---------------------------------------------------------------- halting

    def _halt(self, message: str) -> None:
        console.print(f"[bold yellow]{message}[/bold yellow]")
        self._emit(
            "run_halt",
            message,
            {"goal": self.goal, "cycles": self.cycles, "done": self._is_done()},
        )

    def _finish(self) -> None:
        console.print("[bold green]Goal complete. Halting.[/bold green]")
        record = self.history[-1] if self.history else None
        self._emit(
            "run_complete",
            "Goal complete.",
            {
                "cycles": self.cycles,
                "summary": record.test_result.summary if record is not None else "",
            },
        )
