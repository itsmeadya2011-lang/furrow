"""The Furrow loop: plan -> execute in parallel -> test -> repeat.

The loop is deliberately defensive. Every cycle is wrapped in ``try/except`` and
five independent termination guarantees guarantee that :meth:`Orchestrator.run`
always returns:

1. success -- tests passed and every task completed;
2. ``settings.max_cycles`` (when > 0);
3. stagnation -- the same test-failure signature repeated ``stagnation_limit``
   times in a row;
4. the consecutive-error budget -- planning/testing/worker exceptions;
5. a cooperative :meth:`Orchestrator.stop` call.

``self.goal`` is the immutable objective. Test failures accumulate in
``self.fix_instructions`` instead of overwriting the goal, which is what made the
original loop drift away from what the user asked for.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.text import Text

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TaskStatus, TestResult
from furrow.core.events import Event, EventBus
from furrow.llm import LLMClient
from furrow.state import RunState, StateStore

MAX_FIX_INSTRUCTIONS_CHARS = 4000
"""Upper bound on the accumulated, de-duplicated fix instructions."""


class Orchestrator:
    """Drives one goal through as many cycles as the termination guards allow."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        store: StateStore | None = None,
        events: EventBus | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.console = console if console is not None else Console()
        self.events = events if events is not None else EventBus()
        self.store = (
            store
            if store is not None
            else StateStore(self.settings.resolved_state_path)
        )
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.fix_instructions = ""
        self.state = RunState(goal=goal)
        self.consecutive_errors = 0
        self._running = False
        self._stop_requested = False
        self._test_passed = False
        self._cycle_errors = 0
        self._last_signature = ""
        self._stagnation_count = 0

    # ------------------------------------------------------------------ public

    @property
    def is_running(self) -> bool:
        return self._running

    async def run(self) -> RunState:
        """Run the loop to completion and return the final state."""
        self._running = True
        self._sync_state()
        self._report(
            Event(kind="run_start", message=f"Goal: {self.goal}"),
            rich_message=self._banner(),
        )
        try:
            while not self._should_stop():
                self.cycles += 1
                self._report(
                    Event(kind="cycle_start", cycle=self.cycles, message=f"Cycle {self.cycles}"),
                    rich_message=Text(f"═══ Cycle {self.cycles} ═══", style="bold cyan"),
                )
                await self._run_cycle()
        except asyncio.CancelledError:
            self._halt("run cancelled")
            self._running = False
            self._finish()
            raise
        except Exception as exc:  # noqa: BLE001 - the loop must always return
            self._record_error(f"{type(exc).__name__}: {exc}")
            self._halt(f"unexpected error: {type(exc).__name__}: {exc}")
        self._running = False
        self._finish()
        return self.state

    async def stop(self) -> None:
        """Ask the loop to stop at the next safe point (cooperative)."""
        self._stop_requested = True
        self._report(
            Event(kind="notice", message="stop requested"),
            rich_message=Text("Stop requested.", style="yellow"),
        )

    # --------------------------------------------------------------- the cycle

    async def _run_cycle(self) -> None:
        """Run one cycle and fold its outcome into the loop-wide guards."""
        self._cycle_errors = 0
        test_result: TestResult | None = None
        try:
            test_result = await self._cycle()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad cycle must not kill the run
            self._record_error(f"{type(exc).__name__}: {exc}")
        self._finish_cycle(test_result)

    async def _cycle(self) -> TestResult | None:
        plan = await self._plan()
        if not plan.tasks:
            self._handle_empty_plan()
            return None
        self.tasks = list(plan.tasks)
        self._sync_state()
        await self._execute_tasks()
        return await self._test()

    async def _plan(self) -> Plan:
        with Status("[bold yellow]Planning...", console=self.console):
            plan = await self.planner.plan(self.goal, context=self._context())
        self._report(
            Event(
                kind="plan",
                cycle=self.cycles,
                message=plan.rationale,
                data={
                    "tasks": [task.model_dump(mode="json") for task in plan.tasks],
                    "rationale": plan.rationale,
                },
            ),
            rich_message=self._plan_panel(plan),
        )
        return plan

    def _handle_empty_plan(self) -> None:
        """No tasks planned: finish if there is nothing pending, else burn budget."""
        if not self.fix_instructions.strip():
            self.state.done = True
            self._report(
                Event(
                    kind="notice",
                    cycle=self.cycles,
                    message="planner returned no tasks and nothing is left to fix",
                ),
                rich_message=Text("Nothing left to do. Goal complete.", style="green"),
            )
            return
        self._record_error("planner returned no tasks while fixes are still pending")

    async def _execute_tasks(self) -> None:
        semaphore = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))
        coroutines = [self._run_task(task, semaphore) for task in self.tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        self._collect_results(self.tasks, results)
        await self._retry_failed_tasks(semaphore)

    async def _run_task(
        self, task: TaskModel, semaphore: asyncio.Semaphore
    ) -> str | BaseException:
        task.status = TaskStatus.RUNNING
        task.attempts += 1
        self._report(
            Event(
                kind="task_start",
                cycle=self.cycles,
                message=task.description,
                data={"task_id": task.id, "attempt": task.attempts},
            ),
            rich_message=Text(f"▶ {task.id}: {task.description}", style="cyan"),
        )
        async with semaphore:
            worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
            try:
                return await worker.run()
            except Exception as exc:  # noqa: BLE001 - surfaced as a failed task
                return exc

    async def _retry_failed_tasks(self, semaphore: asyncio.Semaphore) -> None:
        """One extra pass for failures that still have attempt budget left."""
        budget = self.settings.max_task_attempts
        if budget <= 1:
            return
        retryable = [
            task
            for task in self.tasks
            if task.status == TaskStatus.FAILED and task.attempts < budget
        ]
        if not retryable:
            return
        self._report(
            Event(
                kind="notice",
                cycle=self.cycles,
                message=f"retrying {len(retryable)} failed task(s)",
                data={"task_ids": [task.id for task in retryable]},
            ),
            rich_message=Text(f"Retrying {len(retryable)} failed task(s)...", style="yellow"),
        )
        coroutines = [self._run_task(task, semaphore) for task in retryable]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        self._collect_results(retryable, results)

    def _collect_results(self, tasks: list[TaskModel], results: list[Any]) -> None:
        for task, result in zip(tasks, results, strict=False):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException):
                task.status = TaskStatus.FAILED
                task.result = self._truncate(f"{type(result).__name__}: {result}")
                self._cycle_errors += 1
                style = "red"
            else:
                task.status = TaskStatus.COMPLETED
                task.result = self._truncate(str(result))
                style = "green"
            label = f"{task.id} {task.status.value}"
            self._report(
                Event(
                    kind="task_end",
                    cycle=self.cycles,
                    message=label,
                    data={
                        "task_id": task.id,
                        "status": task.status.value,
                        "attempt": task.attempts,
                        "result": task.result,
                    },
                ),
                rich_message=Text(label, style=style),
            )

    async def _test(self) -> TestResult:
        tester = TesterAgent(client=self.client, settings=self.settings)
        with Status("[bold yellow]Testing...", console=self.console):
            result = await tester.run(self.goal, self.tasks, context=self._context())
        self._report(
            Event(
                kind="test_result",
                cycle=self.cycles,
                message=result.summary,
                data=result.model_dump(mode="json"),
            ),
            rich_message=self._test_panel(result),
        )
        if not result.passed:
            self._merge_failures(result.failures)
        return result

    # ----------------------------------------------------------------- guards

    def _should_stop(self) -> bool:
        if self._stop_requested:
            self._halt("stop requested by user")
            return True
        if self.state.done or self.state.halted_reason:
            return True
        limit = self.settings.max_cycles
        if limit > 0 and self.cycles >= limit:
            self._halt("max_cycles reached")
            return True
        return False

    def _finish_cycle(self, test_result: TestResult | None) -> None:
        if test_result is not None:
            self._test_passed = test_result.passed
            if test_result.passed:
                self._stagnation_count = 0
                self._last_signature = ""
            else:
                self._note_stagnation(test_result.failures)
        if self._cycle_errors > 0:
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0
        self._append_history(test_result)
        self._sync_state()
        if self._is_done():
            self.state.done = True
        error_limit = self.settings.max_consecutive_errors
        if error_limit > 0 and self.consecutive_errors >= error_limit:
            self._halt(
                f"too many consecutive errors ({self.consecutive_errors}); "
                "check the LLM provider configuration"
            )
        self._persist()
        self._report(
            Event(
                kind="cycle_end",
                cycle=self.cycles,
                message=self._cycle_summary(test_result),
                data={
                    "passed": self._test_passed,
                    "failures": list(test_result.failures) if test_result else [],
                    "consecutive_errors": self.consecutive_errors,
                    "task_statuses": {task.id: task.status.value for task in self.tasks},
                },
            ),
        )

    def _note_stagnation(self, failures: list[str]) -> None:
        digest = hashlib.sha256(
            "|".join(sorted(str(item).strip() for item in failures)).encode("utf-8")
        ).hexdigest()[:16]
        if digest == self._last_signature:
            self._stagnation_count += 1
        else:
            self._last_signature = digest
            self._stagnation_count = 1
        limit = self.settings.stagnation_limit
        if limit > 0 and self._stagnation_count >= limit:
            self._halt("no progress on repeated test failures")

    def _is_done(self) -> bool:
        """True once tests passed *and* every task completed."""
        if self.state.done:
            return True
        if not self.tasks:
            return False
        if any(task.status != TaskStatus.COMPLETED for task in self.tasks):
            return False
        return self._test_passed

    def _halt(self, reason: str) -> None:
        if not self.state.halted_reason:
            self.state.halted_reason = reason
        self._report(
            Event(
                kind="notice",
                cycle=self.cycles,
                message=reason,
                data={"halted": True},
            ),
            rich_message=Text(f"Halting: {reason}", style="bold yellow"),
        )

    # ------------------------------------------------------------ bookkeeping

    def _record_error(self, message: str) -> None:
        self._cycle_errors += 1
        self._report(
            Event(kind="error", cycle=self.cycles, message=message),
            rich_message=Text(f"Error: {message}", style="bold red"),
        )

    def _merge_failures(self, failures: list[str]) -> None:
        if not failures:
            return
        lines = [line for line in self.fix_instructions.splitlines() if line.strip()]
        seen = {line.strip() for line in lines}
        for failure in failures:
            for raw in str(failure).splitlines():
                key = raw.strip()
                if key and key not in seen:
                    seen.add(key)
                    lines.append(key)
        merged = "\n".join(lines)
        if len(merged) > MAX_FIX_INSTRUCTIONS_CHARS:
            merged = (
                "[earlier fix instructions truncated]\n"
                + merged[-MAX_FIX_INSTRUCTIONS_CHARS:]
            )
        self.fix_instructions = merged
        self.state.fix_instructions = merged

    def _append_history(self, test_result: TestResult | None) -> None:
        self.state.history.append(
            {
                "cycle": self.cycles,
                "passed": test_result.passed if test_result is not None else False,
                "summary": (
                    test_result.summary
                    if test_result is not None
                    else "cycle aborted before tests ran"
                ),
                "failures": list(test_result.failures) if test_result is not None else [],
                "task_statuses": {task.id: task.status.value for task in self.tasks},
            }
        )

    def _sync_state(self) -> None:
        self.state.goal = self.goal
        self.state.cycles = self.cycles
        self.state.tasks = self.tasks
        self.state.fix_instructions = self.fix_instructions

    def _persist(self) -> None:
        self._sync_state()
        try:
            self.store.save(self.state)
        except OSError as exc:
            self._report(
                Event(kind="error", cycle=self.cycles, message=f"could not save state: {exc}"),
                rich_message=Text(f"Could not persist state: {exc}", style="yellow"),
            )

    def _finish(self) -> None:
        self._sync_state()
        self._persist()
        done = self.state.done
        reason = self.state.halted_reason
        outcome = "goal complete" if done else f"halted ({reason or 'unknown'})"
        self._report(
            Event(
                kind="run_end",
                cycle=self.cycles,
                message=f"{outcome} after {self.cycles} cycle(s)",
                data={
                    "done": done,
                    "halted_reason": reason,
                    "cycles": self.cycles,
                    "history": len(self.state.history),
                },
            ),
            rich_message=Panel(
                Text(outcome, style="bold green" if done else "bold yellow"),
                title="Furrow",
            ),
        )

    def _truncate(self, text: str) -> str:
        limit = self.settings.max_output_chars
        if limit > 0 and len(text) > limit:
            return text[:limit] + "\n...[truncated]"
        return text

    def _context(self) -> str:
        """Compact state summary handed to the planner and the tester."""
        budget = self.settings.max_cycles if self.settings.max_cycles > 0 else "unbounded"
        parts = [f"Cycle {self.cycles} of {budget}."]
        if self.tasks:
            completed = sum(
                1 for task in self.tasks if task.status == TaskStatus.COMPLETED
            )
            parts.append(f"Tasks: {completed}/{len(self.tasks)} completed.")
        else:
            parts.append("No tasks have been executed yet.")
        if self.state.history:
            last = self.state.history[-1]
            failures = last.get("failures") or []
            parts.append(f"Last cycle passed={last.get('passed')}, failures={len(failures)}.")
        if self.fix_instructions.strip():
            parts.append("Outstanding failures to fix:\n" + self.fix_instructions)
        return "\n".join(parts)

    # -------------------------------------------------------------- rendering

    def _report(self, event: Event, rich_message: RenderableType | None = None) -> None:
        """Print (when a rich message is supplied) and emit an event."""
        if rich_message is not None:
            self.console.print(rich_message)
        self.events.emit(event)

    def _banner(self) -> RenderableType:
        budget = self.settings.max_cycles if self.settings.max_cycles > 0 else "unbounded"
        return Panel.fit(
            Group(
                Text("Furrow", style="bold green"),
                Text(f"Goal: {self.goal}"),
                Text(f"Cycle budget: {budget}"),
            ),
            title="Furrow",
        )

    def _plan_panel(self, plan: Plan) -> RenderableType:
        if self.settings.verbose:
            return Panel(
                Pretty(plan.model_dump(mode="json")),
                title=f"Plan (cycle {self.cycles})",
                border_style="blue",
            )
        body = Text("\n".join(f"{task.id}: {task.description}" for task in plan.tasks))
        return Panel(body, title=f"Plan (cycle {self.cycles})", border_style="blue")

    def _test_panel(self, result: TestResult) -> RenderableType:
        style = "green" if result.passed else "red"
        lines = [Text(result.summary)]
        lines.extend(Text(f"• {failure}") for failure in result.failures)
        return Panel(Group(*lines), title="Tests", border_style=style)

    def _cycle_summary(self, test_result: TestResult | None) -> str:
        completed = sum(1 for task in self.tasks if task.status == TaskStatus.COMPLETED)
        if test_result is None:
            return f"{completed}/{len(self.tasks)} tasks completed, tests not reached"
        verdict = "passed" if test_result.passed else "failed"
        return f"{completed}/{len(self.tasks)} tasks completed, tests {verdict}"


__all__ = ["MAX_FIX_INSTRUCTIONS_CHARS", "Orchestrator"]