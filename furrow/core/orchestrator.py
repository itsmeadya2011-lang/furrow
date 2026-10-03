from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.config import settings as global_settings
from furrow.core.events import Event, EventType
from furrow.llm import LLMClient

console = Console()

COMPLETED = "completed"
FAILED = "failed"

#: Consecutive cycles without new tasks or state changes before we give up.
NO_PROGRESS_LIMIT = 3
#: How much of a task result is handed back to the planner as context.
RESULT_EXCERPT = 300
#: How many previous plans are summarised for the planner.
HISTORY_WINDOW = 3


def _accepted(fn: Callable[..., Any], **candidates: Any) -> dict[str, Any]:
    """Return the subset of ``candidates`` keyword arguments that ``fn`` accepts."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(candidates)
    return {name: value for name, value in candidates.items() if name in params}


class Orchestrator:
    """Drives the plan -> execute -> test loop until the goal is met."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_event: Callable[[Event], Awaitable[None] | None] | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or global_settings
        self.on_event = on_event
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        # Every task the planner has ever produced, keyed by id, latest wins.
        self.tasks: dict[str, TaskModel] = {}
        self.history: list[Plan] = []
        self._pending_fix: str | None = None
        self._test_failures: list[str] = []
        self._no_progress_cycles = 0
        self._stop_requested = False

    # ------------------------------------------------------------------ public

    def request_stop(self) -> None:
        """Ask the loop to stop before starting another cycle."""
        self._stop_requested = True

    def state(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the loop state."""
        return {
            "goal": self.goal,
            "cycles": self.cycles,
            "done": self._is_done(),
            "tasks": [task.model_dump() for task in self._get_tasks()],
            "history": [plan.model_dump() for plan in self.history],
        }

    async def run(self, max_cycles: int | None = None) -> None:
        """Run the development loop until the goal is met, stopped, or limited."""
        limit = self._cycle_limit(max_cycles)
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        reason = "Goal complete."
        try:
            while True:
                if self._stop_requested:
                    reason = "Stop requested by caller."
                    break
                if 0 < limit <= self.cycles:
                    reason = f"Reached the cycle limit of {limit}."
                    break
                self.cycles += 1
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                before = self._snapshot()
                await self._emit(EventType.CYCLE_START, f"cycle {self.cycles} started")
                try:
                    await self._cycle()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - one bad cycle must not stop the loop
                    await self._error(f"Cycle {self.cycles} failed: {exc}")
                finally:
                    await self._emit(EventType.CYCLE_END, f"cycle {self.cycles} finished")
                self._track_progress(before)
                if self._is_done():
                    reason = "Goal complete."
                    break
                if self._no_progress_cycles >= NO_PROGRESS_LIMIT:
                    reason = (
                        f"No progress for {self._no_progress_cycles} cycles "
                        "(no new tasks, no state changes)"
                    )
                    await self._error(
                        f"{reason}. The goal may be unreachable; stopping to avoid "
                        "spinning forever."
                    )
                    break
        except asyncio.CancelledError:
            console.print("[yellow]Furrow run cancelled.[/yellow]")
            raise
        console.print(f"[bold green]{reason} Halting.[/bold green]")
        await self._emit(EventType.DONE, reason, self._progress())

    # ------------------------------------------------------------------ cycle

    async def _cycle(self) -> None:
        goal = self._planner_goal()
        context = self._planner_context()
        extra = _accepted(self.planner.plan, context=context)
        if context and not extra:
            goal = f"{goal}\n\n{context}"
        self._pending_fix = None  # consumed by this planner call
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(goal, **extra)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit(EventType.PLAN, plan.rationale, plan.model_dump())
        self.history.append(plan)
        for task in plan.tasks:
            self.tasks[task.id] = task
        if not plan.tasks:
            console.print("[yellow]No tasks planned; nothing to execute.[/yellow]")
        await self._execute(plan.tasks)
        await self._test(plan)

    async def _execute(self, tasks: list[TaskModel]) -> None:
        if not tasks:
            return
        semaphore = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))
        waves = self._waves(tasks)
        for number, wave in enumerate(waves, start=1):
            console.print(
                f"[dim]wave {number}/{len(waves)}: "
                f"{', '.join(task.id for task in wave)}[/dim]"
            )
            with Status("[bold yellow]Executing tasks...", console=console):
                await self._run_wave(wave, semaphore)

    async def _run_wave(self, wave: list[TaskModel], semaphore: asyncio.Semaphore) -> None:
        async def _one(task: TaskModel) -> Any:
            async with semaphore:
                await self._emit(
                    EventType.TASK_START, task.description, {"task": task.model_dump()}
                )
                worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
                return await worker.run()

        results = await asyncio.gather(*(_one(task) for task in wave), return_exceptions=True)
        # `gather` yields exactly one entry per coroutine, so the lengths always match;
        # `strict=True` turns any future divergence into a loud error instead of a
        # silently dropped task.
        for task, result in zip(wave, results, strict=True):
            if isinstance(result, BaseException):
                task.status = FAILED
                task.result = f"{type(result).__name__}: {result}"
                await self._error(f"Task {task.id} failed: {result}")
            else:
                task.status = COMPLETED
                task.result = result if isinstance(result, str) else str(result)
                console.print(f"[green]Task {task.id} completed: {task.description}[/green]")
            await self._emit(
                EventType.TASK_END, task.description, {"task": task.model_dump()}
            )

    async def _test(self, plan: Plan) -> None:
        await self._emit(EventType.TEST_START, "running tests", {"goal": self.goal})
        test_result = await self._run_tester(plan)
        await self._emit(EventType.TEST_END, test_result.summary, test_result.model_dump())
        if test_result.passed:
            self._test_failures = []
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            return
        self._test_failures = list(test_result.failures)
        console.print(f"[red]Tests failed: {test_result.summary}[/red]")
        for failure in test_result.failures:
            console.print(f"  • {failure}")
        self._pending_fix = "Fix failing tests:\n" + "\n".join(test_result.failures)
        console.print("[yellow]Will attempt a fix in the next cycle.[/yellow]")

    async def _run_tester(self, plan: Plan) -> TestResult:
        with Status("[bold yellow]Testing...", console=console):
            return await self.tester.run(self.goal, list(plan.tasks))

    # ------------------------------------------------------------- scheduling

    def _waves(self, tasks: list[TaskModel]) -> list[list[TaskModel]]:
        """Split a plan into dependency-ordered waves, preserving plan order."""
        remaining = list(enumerate(tasks))
        satisfied: set[str] = set()
        waves: list[list[TaskModel]] = []
        while remaining:
            plan_ids = {task.id for _, task in remaining}
            ready = [
                index
                for index, task in remaining
                if not ({dep for dep in task.dependencies if dep in plan_ids} - satisfied)
            ]
            if not ready:
                # Dependency cycle: break it by running the rest as one final wave.
                ready = [index for index, _ in remaining]
            chosen = set(ready)
            wave = [task for index, task in remaining if index in chosen]
            waves.append(wave)
            satisfied.update(task.id for task in wave)
            remaining = [item for item in remaining if item[0] not in chosen]
        return waves

    # ------------------------------------------------------------------ state

    def _is_done(self) -> bool:
        """True when no work is outstanding and no test failure is left to fix."""
        if self._test_failures:
            return False
        tasks = self._get_tasks()
        if not tasks:
            # Only a planner that ran and found nothing can end the loop.
            return bool(self.history)
        return all(task.status == COMPLETED for task in tasks)

    def _get_tasks(self) -> list[TaskModel]:
        """Return every task seen so far, keyed by id in discovery order."""
        return list(self.tasks.values())

    def _planner_goal(self) -> str:
        """Build the planner goal, folding in any pending test-failure follow-up."""
        if self._pending_fix:
            return f"{self.goal}\n\n{self._pending_fix}"
        return self.goal

    def _planner_context(self) -> str:
        """Summarise previous cycles so consecutive plans build on each other."""
        if not self.history and not self._test_failures:
            return ""
        lines = ["Context from previous cycles:"]
        for plan in self.history[-HISTORY_WINDOW:]:
            lines.append(f"Plan rationale: {plan.rationale}")
            for task in plan.tasks:
                known = self.tasks.get(task.id, task)
                line = f"  - [{task.id}] {task.description} -> {known.status}"
                result = (known.result or "").strip().replace("\n", " ")
                if result:
                    line += f" | result: {result[:RESULT_EXCERPT]}"
                lines.append(line)
        if self._test_failures:
            lines.append("Failing tests that must be fixed:")
            lines.extend(f"  - {failure}" for failure in self._test_failures)
        lines.append("Continue from this state and do not repeat completed work.")
        return "\n".join(lines)

    def _snapshot(self) -> tuple[set[str], dict[str, tuple[str, Any]]]:
        """Capture task ids and states so progress can be measured per cycle."""
        return set(self.tasks), {
            task_id: (task.status, task.result) for task_id, task in self.tasks.items()
        }

    def _track_progress(self, before: tuple[set[str], dict[str, tuple[str, Any]]]) -> None:
        """Reset or extend the consecutive no-progress cycle counter."""
        known_ids, states = before
        if set(self.tasks) - known_ids:
            self._no_progress_cycles = 0
            return
        if any(
            states.get(task_id) != (task.status, task.result)
            for task_id, task in self.tasks.items()
        ):
            self._no_progress_cycles = 0
            return
        self._no_progress_cycles += 1

    def _cycle_limit(self, override: int | None) -> int:
        """Resolve the cycle budget; zero or negative means unlimited."""
        if override is not None:
            return override
        return self.settings.max_cycles

    def _progress(self) -> dict[str, Any]:
        """Return the run summary attached to the DONE event."""
        return {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": {task_id: task.status for task_id, task in self.tasks.items()},
            "test_failures": list(self._test_failures),
            "no_progress_cycles": self._no_progress_cycles,
            "stop_requested": self._stop_requested,
            "done": self._is_done(),
        }

    # ----------------------------------------------------------------- events

    async def _emit(
        self, event_type: EventType, message: str = "", data: dict[str, Any] | None = None
    ) -> None:
        """Emit an event, never letting a faulty callback break the loop."""
        if self.on_event is None:
            return
        event = Event(type=event_type, cycle=self.cycles, message=message, data=data or {})
        try:
            result = self.on_event(event)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 - a faulty callback must not break the loop
            console.print(f"[yellow]Event callback for {event_type.value} failed: {exc}[/yellow]")

    async def _error(self, message: str) -> None:
        """Report a recoverable problem on the console and as an ERROR event."""
        console.print(f"[red]{message}[/red]")
        await self._emit(EventType.ERROR, message, {"cycle": self.cycles})