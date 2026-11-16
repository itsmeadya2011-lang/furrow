"""Furrow Orchestrator - coordinates the autonomous coding loop.

The orchestrator runs cycles of: plan -> execute -> test -> repeat
until the goal is achieved, tests pass, or a stopping condition is met.
"""

from __future__ import annotations

import asyncio
from typing import Any

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.config import RunResult, Settings, TaskModel, Usage
from furrow.config import settings as global_settings
from furrow.core.cycle_manager import CycleManager
from furrow.core.events import Event, EventBus, EventType
from furrow.core.state import HistoryEntry, RunState, StateStore
from furrow.llm import LLMClient

_UNSET = object()

_MAX_TEXT = 120


class Orchestrator:
    """Main orchestrator for the Furrow autonomous coding agent."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        state_store: StateStore | None = None,
        event_bus: EventBus | None = None,
        resume: bool = False,
    ) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)
        self.goal = goal
        self.original_goal = goal
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.event_bus = event_bus if event_bus is not None else EventBus()
        self.state_store = (
            state_store if state_store is not None else StateStore(self.settings.state_path())
        )
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings, workspace=self.settings.workspace)
        self.history: list[HistoryEntry] = []
        self._feedback = ""
        self._active_ids: set[str] = set()
        self._prev_sig: str = ""
        self._no_progress = 0
        self._stopped_reason = ""
        self._resume = resume
        self._last_test_passed: bool | None = None
        self._tests_skipped: bool = False
        self._console_obj: Any = _UNSET
        self._resume_state_missing = False
        self._usage: Usage = Usage()
        self._usage_seen: Usage = Usage()
        self._tests_run: int = 0

        if resume:
            # The state file is read exactly once, here: `run()` must not load it
            # again, otherwise the values restored at start-up would be silently
            # overwritten by a second (possibly different) read.
            st = self.state_store.load()
            if st is not None:
                self.goal = st.goal
                self.original_goal = st.original_goal
                self.cycles = st.cycles
                self.tasks = list(st.tasks)
                self.history = list(st.history)
                self._usage = st.usage.model_copy()
                # A resumed run starts a fresh client whose counter starts at
                # zero, so the baseline for the per-cycle delta is the client's
                # CURRENT usage, not the persisted total.
                self._usage_seen = self._client_usage()
                # Restore no-progress guard state from persisted run state
                try:
                    self._no_progress = int(getattr(st, "no_progress", 0) or 0)
                except Exception:
                    self._no_progress = 0
                try:
                    self._prev_sig = str(getattr(st, "prev_sig", "") or "")
                except Exception:
                    self._prev_sig = ""
                try:
                    self._tests_run = int(getattr(st, "tests_run", 0) or 0)
                except Exception:
                    self._tests_run = 0
            else:
                self._resume_state_missing = True

    def _client_usage(self) -> Usage:
        """Cumulative usage of the LLM client (zero when the client does not report it)."""
        raw = getattr(self.client, "total_usage", None)
        if raw is None:
            return Usage()
        return Usage(
            input_tokens=int(getattr(raw, "input_tokens", 0) or 0),
            output_tokens=int(getattr(raw, "output_tokens", 0) or 0),
            calls=int(getattr(raw, "calls", 0) or 0),
            cost_usd=float(getattr(raw, "cost_usd", 0.0) or 0.0),
        )

    def status(self) -> dict[str, Any]:
        """Return a snapshot of the current run."""
        return {
            "goal": self.goal,
            "original_goal": self.original_goal,
            "cycles": self.cycles,
            "stopped_reason": self._stopped_reason,
            "tasks": [t.model_dump() for t in self.tasks],
            "usage": self._usage.model_dump(),
        }

    def _console(self) -> Any:
        """Return a rich Console, or None when rich is unavailable."""
        if self._console_obj is _UNSET:
            try:
                from rich.console import Console

                self._console_obj = Console()
            except ImportError:  # pragma: no cover - rich is a declared dependency
                self._console_obj = None
        return self._console_obj

    def _say(self, message: str) -> None:
        """Print a progress message; a broken console must never break the run."""
        console = self._console()
        if console is None:
            return
        try:
            console.print(message)
        except Exception:  # pragma: no cover - printing must never break a run
            return

    @staticmethod
    def _one_line(text: str) -> str:
        """Collapse whitespace so *text* fits on a single terminal line."""
        return " ".join(text.split())

    @classmethod
    def _clip(cls, text: str, limit: int = _MAX_TEXT) -> str:
        """Return *text* on a single line, truncated to at most *limit* characters."""
        flat = cls._one_line(text)
        if len(flat) <= limit:
            return flat
        return flat[: limit - 1].rstrip() + "…"

    @staticmethod
    def _escape(text: str) -> str:
        """Escape rich markup so untrusted text is printed verbatim."""
        try:
            from rich.markup import escape
        except ImportError:  # pragma: no cover - rich is a declared dependency
            return text.replace("[", "\\[")
        return str(escape(text))

    async def run(self) -> RunResult:
        if self._resume and self._resume_state_missing:
            # The state file was already looked up in __init__; it was missing.
            self._resume_state_missing = False
            self.event_bus.emit(Event(type=EventType.LOG, message="no state file found, starting fresh"))

        self._say(f"[bold]Goal:[/bold] {self._escape(self._clip(self.original_goal))}")

        while True:
            self.cycles += 1

            cycle_manager = CycleManager(
                settings=self.settings,
                client=self.client,
                goal=self.goal,
                original_goal=self.original_goal,
                tasks=self.tasks,
                planner=self.planner,
                tester=self.tester,
                event_bus=self.event_bus,
                state_store=self.state_store,
                history=self.history,
                usage=self._usage,
                usage_seen=self._usage_seen,
                tests_run=self._tests_run,
                no_progress=self._no_progress,
                prev_sig=self._prev_sig,
                last_test_passed=self._last_test_passed,
                tests_skipped=self._tests_skipped,
                feedback=self._feedback,
                stopped_reason=self._stopped_reason,
                _say=self._say,
                _escape=self._escape,
                _one_line=self._one_line,
                _clip=self._clip,
                _client_usage=self._client_usage,
            )

            continue_loop = await cycle_manager.run_cycle(self.cycles)

            # Update state from cycle manager
            self.goal = cycle_manager.goal
            self._feedback = cycle_manager.feedback
            self._stopped_reason = cycle_manager.stopped_reason
            self._last_test_passed = cycle_manager.last_test_passed
            self._tests_skipped = cycle_manager.tests_skipped
            self._usage = cycle_manager.usage
            self._usage_seen = cycle_manager._usage_seen  # type: ignore
            self._tests_run = cycle_manager.tests_run
            self._no_progress = cycle_manager._no_progress  # type: ignore
            self._prev_sig = cycle_manager._prev_sig  # type: ignore
            self._active_ids = self.planner.skip_ids

            if not continue_loop:
                break

        if not self._stopped_reason:
            if self.settings.max_cycles and self.cycles >= self.settings.max_cycles:
                self._stopped_reason = "max_cycles reached"
            else:
                self._stopped_reason = "goal complete"

        self._say(f"Stopped: {self._escape(self._one_line(self._stopped_reason))}")

        active = (
            [t for t in self.tasks if t.id in self._active_ids] if self._active_ids else self.tasks
        )
        completed = sum(1 for t in active if t.status == "completed")
        failed = sum(1 for t in active if t.status == "failed")
        blocked = sum(1 for t in active if t.status == "blocked")
        files_changed = sorted(
            {f for t in active for f in (getattr(t, "files_changed", None) or [])}
        )
        return RunResult(
            passed=bool(self._last_test_passed),
            cycles=self.cycles,
            tasks_completed=completed,
            tasks_failed=failed,
            tasks_blocked=blocked,
            tests_run=int(getattr(self, "_tests_run", 0) or 0),
            summary=f"stopped: {self._stopped_reason}",
            stopped_reason=self._stopped_reason,
            skipped=bool(self._tests_skipped),
            usage=self._usage,
            files_changed=files_changed,
        )