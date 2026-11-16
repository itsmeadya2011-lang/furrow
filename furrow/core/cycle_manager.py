"""Cycle management for the Furrow orchestrator.

Handles the main orchestration loop: plan -> execute -> test -> repeat.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.config import Plan, RunResult, Settings, TaskModel, Usage
from furrow.core.events import Event, EventBus, EventType
from furrow.core.state import HistoryEntry, RunState, StateStore


class CycleManager:
    """Manages the orchestration cycles."""

    def __init__(
        self,
        settings: Settings,
        client: Any,
        goal: str,
        original_goal: str,
        tasks: list[TaskModel],
        planner: PlannerAgent,
        tester: TesterAgent,
        event_bus: EventBus,
        state_store: StateStore,
        history: list[HistoryEntry],
        usage: Usage,
        usage_seen: Usage,
        tests_run: int,
        no_progress: int,
        prev_sig: str,
        last_test_passed: bool | None,
        tests_skipped: bool,
        feedback: str,
        stopped_reason: str,
        _say: Any,
        _escape: Any,
        _one_line: Any,
        _clip: Any,
        _client_usage: Any,
    ) -> None:
        self.settings = settings
        self.client = client
        self.goal = goal
        self.original_goal = original_goal
        self.tasks = tasks
        self.planner = planner
        self.tester = tester
        self.event_bus = event_bus
        self.state_store = state_store
        self.history = history
        self._usage = usage
        self._usage_seen = usage_seen
        self._tests_run = tests_run
        self._no_progress = no_progress
        self._prev_sig = prev_sig
        self._last_test_passed = last_test_passed
        self._tests_skipped = tests_skipped
        self._feedback = feedback
        self._stopped_reason = stopped_reason
        self._say = _say
        self._escape = _escape
        self._one_line = _one_line
        self._clip = _clip
        self._client_usage = _client_usage

    @property
    def stopped_reason(self) -> str:
        return self._stopped_reason

    @property
    def feedback(self) -> str:
        return self._feedback

    @property
    def goal(self) -> str:
        return self._goal

    @goal.setter
    def goal(self, value: str) -> None:
        self._goal = value

    @property
    def last_test_passed(self) -> bool | None:
        return self._last_test_passed

    @property
    def tests_skipped(self) -> bool:
        return self._tests_skipped

    @property
    def usage(self) -> Usage:
        return self._usage

    @property
    def tests_run(self) -> int:
        return self._tests_run

    async def run_cycle(self, cycle: int) -> bool:
        """Run a single orchestration cycle.

        Returns True if the loop should continue, False if it should stop.
        """
        if self.settings.max_cycles and cycle >= self.settings.max_cycles:
            self._stopped_reason = "max_cycles reached"
            return False

        self._say(f"[bold]\u2550\u2550\u2550 Cycle {cycle} \u2550\u2550\u2550[/bold]")
        self.event_bus.emit(Event(type=EventType.CYCLE_STARTED, data={"cycle": cycle}))

        # Plan
        try:
            plan = await self.planner.plan(self.goal, context=self._feedback)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.event_bus.emit(Event(type=EventType.ERROR, message=str(e)))
            self._stopped_reason = f"planner failed: {e}"
            return False

        self.event_bus.emit(
            Event(type=EventType.PLAN_CREATED, data={"rationale": plan.rationale, "tasks": len(plan.tasks)})
        )
        planned = len(plan.tasks)
        unit = "task" if planned == 1 else "tasks"
        self._say(f"Plan: {self._escape(self._clip(plan.rationale))} ({planned} {unit})")

        if not plan.tasks:
            self.event_bus.emit(Event(type=EventType.LOG, message="no tasks planned - goal complete"))
            self._stopped_reason = "goal complete"
            return False

        return await self._execute_and_test(plan, cycle)

    async def _execute_and_test(self, plan: Plan, cycle: int) -> bool:
        """Execute the plan and run tests."""
        # Merge plan (updates self.tasks)
        self.planner.merge_plan(self.tasks, plan)
        self._active_ids = {t.id for t in plan.tasks}

        # Execute
        from furrow.core.task_executor import TaskExecutor

        executor = TaskExecutor(
            settings=self.settings,
            client=self.client,
            workspace=self.settings.workspace,
            original_goal=self.original_goal,
            event_bus=self.event_bus,
        )
        try:
            waves = self.planner.plan_waves(plan.tasks)
            await executor.execute(
                plan.tasks,
                waves,
                self.planner.skip_ids,
                self._escape,
                self._one_line,
                self._say,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.event_bus.emit(Event(type=EventType.ERROR, message=str(e)))
            self._stopped_reason = f"execute failed: {e}"
            return False

        # Test
        self.event_bus.emit(Event(type=EventType.TEST_STARTED, data={}))
        try:
            test_result = await self.tester.run(self.original_goal, self.tasks)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.event_bus.emit(Event(type=EventType.ERROR, message=str(e)))
            self._stopped_reason = f"tester failed: {e}"
            return False

        self._tests_run += 1
        skipped = bool(getattr(test_result, "skipped", False))
        self.event_bus.emit(
            Event(
                type=EventType.TEST_FINISHED,
                data={"passed": test_result.passed, "summary": test_result.summary},
            )
        )
        summary = self._escape(self._one_line(test_result.summary))
        if test_result.passed:
            self._say(f"[green]Tests passed: {summary}[/green]")
        else:
            self._say(f"[red]Tests failed: {summary}[/red]")

        self._last_test_passed = test_result.passed
        self._tests_skipped = skipped

        if skipped:
            # A skipped run (e.g. no test runner detected) is not a failure:
            # keep the original goal and drop the stale feedback so the loop
            # does not chase a test runner that does not exist.
            self.goal = self.original_goal
            self._feedback = ""
        elif not test_result.passed:
            failures = [f for f in test_result.failures if f and f.strip()]
            if not failures:
                failures = [
                    test_result.summary
                    or "the test suite failed without a detailed report"
                ]
            self._feedback = "\n".join(failures)
            self.goal = (
                f"{self.original_goal}\n\n"
                "The previous cycle left failing tests. Fix them:\n"
                f"{self._feedback}"
            )

        # Update usage
        cumulative = self._client_usage()
        cycle_usage = Usage(
            input_tokens=max(0, cumulative.input_tokens - self._usage_seen.input_tokens),
            output_tokens=max(0, cumulative.output_tokens - self._usage_seen.output_tokens),
            calls=max(0, cumulative.calls - self._usage_seen.calls),
            cost_usd=max(0.0, cumulative.cost_usd - self._usage_seen.cost_usd),
        )
        self._usage.add(cycle_usage)
        self._usage_seen = cumulative

        # Record history
        self.history.append(
            HistoryEntry(
                cycle=cycle,
                goal=self.goal,
                plan_rationale=plan.rationale,
                task_summaries=[t.id for t in self.tasks],
                tests_passed=test_result.passed,
                test_summary=test_result.summary,
                timestamp=datetime.now(timezone.utc).isoformat(),
                usage=cycle_usage,
            )
        )

        self.event_bus.emit(Event(type=EventType.CYCLE_FINISHED, data={"cycle": cycle}))

        # No-progress guard
        sig = "|".join(f"{t.id}:{t.description}" for t in plan.tasks)
        # A skipped run (no test runner) reports passed=True but gives the
        # loop no verification signal, so it must trip the no-progress
        # guard as well - otherwise the loop never stops.
        no_signal = (not test_result.passed) or skipped
        guard_tripped = False
        if no_signal and sig == self._prev_sig:
            self._no_progress += 1
            if self._no_progress >= 2:
                self._stopped_reason = "no test runner detected" if skipped else "no progress"
                guard_tripped = True
        else:
            if sig != self._prev_sig or (test_result.passed and not skipped):
                self._no_progress = 0
            self._prev_sig = sig

        # Save state
        state = RunState(
            goal=self.goal,
            original_goal=self.original_goal,
            cycles=cycle,
            tasks=self.tasks,
            history=self.history,
            usage=self._usage,
            no_progress=self._no_progress,
            prev_sig=self._prev_sig,
            tests_run=self._tests_run,
            last_passed=test_result.passed,
            last_skipped=skipped,
            last_summary=test_result.summary,
        )
        try:
            self.state_store.save(state)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.event_bus.emit(
                Event(type=EventType.ERROR, message=f"could not save state to {self.state_store.path}: {e}")
            )
        else:
            self.event_bus.emit(Event(type=EventType.STATE_SAVED, data={"path": str(self.state_store.path)}))

        if guard_tripped:
            return False

        return True

    def _context(self) -> str:
        if self._feedback:
            return self._feedback
        return ""