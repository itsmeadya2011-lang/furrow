from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, RunResult, Settings, TaskModel, Usage
from furrow.config import settings as global_settings
from furrow.core.events import Event, EventBus, EventType
from furrow.core.state import HistoryEntry, RunState, StateStore
from furrow.llm import LLMClient

_UNSET = object()

_MAX_TEXT = 120


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None,
                 settings: Settings | None = None, state_store: StateStore | None = None,
                 event_bus: EventBus | None = None, resume: bool = False) -> None:
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
        self.tester = TesterAgent(client=self.client, settings=self.settings,
                                 workspace=self.settings.workspace)
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
        self._skip_ids: set[str] = set()
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
            self.event_bus.emit(Event(type=EventType.LOG,
                                      message="no state file found, starting fresh"))

        self._say(f"[bold]Goal:[/bold] {self._escape(self._clip(self.original_goal))}")

        while True:
            if self.settings.max_cycles and self.cycles >= self.settings.max_cycles:
                self._stopped_reason = "max_cycles reached"
                break
            self.cycles += 1
            self._say(f"[bold]═══ Cycle {self.cycles} ═══[/bold]")
            self.event_bus.emit(Event(type=EventType.CYCLE_STARTED, data={"cycle": self.cycles}))
            try:
                plan = await self.planner.plan(self.goal, context=self._context())
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.event_bus.emit(Event(type=EventType.ERROR, message=str(e)))
                self._stopped_reason = f"planner failed: {e}"
                self.cycles -= 1
                break
            self.event_bus.emit(Event(type=EventType.PLAN_CREATED,
                                    data={"rationale": plan.rationale, "tasks": len(plan.tasks)}))
            planned = len(plan.tasks)
            unit = "task" if planned == 1 else "tasks"
            self._say(f"Plan: {self._escape(self._clip(plan.rationale))} ({planned} {unit})")
            if not plan.tasks:
                self.event_bus.emit(
                    Event(type=EventType.LOG, message="no tasks planned - goal complete")
                )
                self._stopped_reason = "goal complete"
                break
            self._merge_plan(plan)
            try:
                await self._execute(plan)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.event_bus.emit(Event(type=EventType.ERROR, message=str(e)))
                self._stopped_reason = f"execute failed: {e}"
                self.cycles -= 1
                break
            try:
                test_result = await self.tester.run(self.original_goal, self.tasks)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.event_bus.emit(Event(type=EventType.ERROR, message=str(e)))
                self._stopped_reason = f"tester failed: {e}"
                self.cycles -= 1
                break
            self._tests_run += 1
            skipped = bool(getattr(test_result, "skipped", False))
            self.event_bus.emit(Event(type=EventType.TEST_STARTED, data={}))
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
                self.goal = (f"{self.original_goal}\n\n"
                             "The previous cycle left failing tests. Fix them:\n"
                             f"{self._feedback}")
            cumulative = self._client_usage()
            cycle_usage = Usage(
                input_tokens=max(0, cumulative.input_tokens - self._usage_seen.input_tokens),
                output_tokens=max(0, cumulative.output_tokens - self._usage_seen.output_tokens),
                calls=max(0, cumulative.calls - self._usage_seen.calls),
                cost_usd=max(0.0, cumulative.cost_usd - self._usage_seen.cost_usd),
            )
            self._usage.add(cycle_usage)
            self._usage_seen = cumulative
            self.history.append(HistoryEntry(
                cycle=self.cycles, goal=self.goal, plan_rationale=plan.rationale,
                task_summaries=[t.id for t in self.tasks],
                tests_passed=test_result.passed, test_summary=test_result.summary,
                timestamp=datetime.now(timezone.utc).isoformat(),
                usage=cycle_usage,
            ))
            self.event_bus.emit(Event(type=EventType.CYCLE_FINISHED, data={"cycle": self.cycles}))
            sig = "|".join(f"{t.id}:{t.description}" for t in plan.tasks)
            # A skipped run (no test runner) reports passed=True but gives the
            # loop no verification signal, so it must trip the no-progress
            # guard as well - otherwise the loop never stops.
            no_signal = (not test_result.passed) or skipped
            guard_tripped = False
            if no_signal and sig == self._prev_sig:
                self._no_progress += 1
                if self._no_progress >= 2:
                    self._stopped_reason = (
                        "no test runner detected" if skipped else "no progress"
                    )
                    guard_tripped = True
            else:
                if sig != self._prev_sig or (test_result.passed and not skipped):
                    self._no_progress = 0
                self._prev_sig = sig

            state = RunState(goal=self.goal, original_goal=self.original_goal,
                             cycles=self.cycles, tasks=self.tasks, history=self.history,
                             usage=self._usage, no_progress=self._no_progress,
                             prev_sig=self._prev_sig, tests_run=self._tests_run,
                             last_passed=test_result.passed, last_skipped=skipped,
                             last_summary=test_result.summary)
            try:
                self.state_store.save(state)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # The state was NOT written: report the failure (including the
                # path that could not be written) and do NOT emit STATE_SAVED,
                # which would tell consumers the state was saved when it was not.
                self.event_bus.emit(
                    Event(type=EventType.ERROR,
                         message=f"could not save state to {self.state_store.path}: {e}")
                )
            else:
                self.event_bus.emit(Event(type=EventType.STATE_SAVED,
                                        data={"path": str(self.state_store.path)}))
            if guard_tripped:
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

    def _context(self) -> str:
        if self._feedback:
            return self._feedback
        return ""

    def _merge_plan(self, plan: Plan) -> None:
        self._skip_ids = set()
        by_id = {t.id: t for t in self.tasks}
        max_retries = max(0, self.settings.max_task_retries)
        for index, pt in enumerate(plan.tasks):
            old = by_id.get(pt.id)
            if old is not None:
                old_desc = old.description
                old_files = list(old.files)
                # Preserve terminal states; only retry-eligible failed tasks may
                # be reset to pending. Blocked tasks must remain blocked so that
                # retry budgets and dependency outcomes are not lost across cycles.
                if old.status == "completed":
                    if old_desc == pt.description and old_files == pt.files:
                        pt.status = "completed"
                        self._skip_ids.add(pt.id)
                    else:
                        pt.status = "pending"
                elif old.status == "failed":
                    if old.attempts <= max_retries:
                        pt.status = "pending"
                    else:
                        pt.status = "failed"
                elif old.status == "blocked":
                    pt.status = "blocked"
                else:
                    # pending / running / skipped: reset to pending so the task
                    # is re-attempted in this cycle.
                    pt.status = "pending"
                old.description = pt.description
                old.files = pt.files
                old.dependencies = pt.dependencies
                old.status = pt.status
                # A freshly planned task carries result=None and error=None,
                # so never let that wipe a result or error already recorded by
                # a previous cycle. Only adopt non-None values from the plan.
                if pt.result is not None:
                    old.result = pt.result
                if pt.error is not None:
                    old.error = pt.error
                plan.tasks[index] = old
            else:
                self.tasks.append(pt)
        self._active_ids = {t.id for t in plan.tasks}

    def _plan_waves(self, tasks: list[TaskModel]) -> list[list[TaskModel]]:
        ids = {t.id for t in tasks}
        waves: list[list[TaskModel]] = []
        done: set[str] = set()
        remaining = list(tasks)
        while remaining:
            wave = [t for t in remaining if all(d in done or d not in ids for d in t.dependencies)]
            if not wave:
                for t in remaining:
                    t.status = "blocked"
                break
            waves.append(wave)
            for t in wave:
                done.add(t.id)
            remaining = [t for t in remaining if t not in wave]
        return waves

    async def _execute(self, plan: Plan) -> None:
        waves = self._plan_waves(plan.tasks)
        sem = asyncio.Semaphore(self.settings.max_parallel_tasks)
        max_retries = max(0, self.settings.max_task_retries)
        workers: dict[str, WorkerAgent] = {}

        def worker_for(task: TaskModel) -> WorkerAgent:
            """Return the worker for *task*, creating it once so retries reuse it."""
            worker = workers.get(task.id)
            if worker is None:
                worker = WorkerAgent(task=task, client=self.client, settings=self.settings,
                                     workspace=self.settings.workspace, goal=self.original_goal)
                workers[task.id] = worker
            return worker

        async def run_one(task: TaskModel) -> None:
            if task.id in self._skip_ids:
                self.event_bus.emit(Event(type=EventType.LOG,
                                        message=f"task {task.id} already completed - skipping"))
                return
            async with sem:
                task.status = "running"
                task.started_at = datetime.now(timezone.utc).isoformat()
                self.event_bus.emit(Event(type=EventType.TASK_STARTED, data={"task": task.id}))
                try:
                    result = await worker_for(task).run()
                except asyncio.CancelledError:
                    raise  # cancellation is not a task failure
                except Exception as exc:
                    # A raised exception is handled exactly like a worker that
                    # returned success=False: it consumes one attempt and follows
                    # the same retry / terminal-failure path.
                    task.attempts += 1
                    task.error = f"{type(exc).__name__}: {exc}"
                    task.finished_at = datetime.now(timezone.utc).isoformat()
                    task.status = "pending" if task.attempts <= max_retries else "failed"
                    self.event_bus.emit(
                        Event(
                            type=EventType.ERROR,
                            message=f"task {task.id} raised: {task.error}",
                            data={"task": task.id, "error": task.error,
                                  "attempts": task.attempts},
                        )
                    )
                    self.event_bus.emit(
                        Event(
                            type=EventType.TASK_FINISHED,
                            data={
                                "task": task.id,
                                "success": False,
                                "summary": str(task.error or ""),
                            },
                        )
                    )
                    task_id = self._escape(task.id)
                    detail = self._escape(self._one_line(task.error))
                    if task.status == "pending":
                        self._say(f"[yellow]Task {task_id} failed: {detail} - retrying[/yellow]")
                        self.event_bus.emit(
                            Event(type=EventType.LOG,
                                 message=f"task {task.id} failed, retrying "
                                         f"(attempt {task.attempts} of {max_retries + 1})")
                        )
                    else:
                        self._say(f"[red]Task {task_id} failed: {detail}[/red]")
                    return
                try:
                    success = bool(getattr(result, "success", False))
                    task.result = getattr(result, "summary", None)
                    task.error = getattr(result, "error", None)
                    task.attempts += 1
                    task.files_changed = sorted(
                        {str(p) for p in (getattr(result, "files_changed", None) or [])}
                    )
                    task.finished_at = datetime.now(timezone.utc).isoformat()
                    if success:
                        task.status = "completed"
                    elif task.attempts <= max_retries:
                        task.status = "pending"
                    else:
                        task.status = "failed"
                    self.event_bus.emit(
                        Event(
                            type=EventType.TASK_FINISHED,
                            data={
                                "task": task.id,
                                "success": task.status == "completed",
                                "summary": str(task.result or ""),
                            },
                        )
                    )
                    task_id = self._escape(task.id)
                    detail = self._escape(self._one_line(task.error or "unknown error"))
                    if task.status == "completed":
                        self._say(f"[green]Task {task_id} completed[/green]")
                    elif task.status == "pending":
                        self._say(f"[yellow]Task {task_id} failed: {detail} - retrying[/yellow]")
                        self.event_bus.emit(
                            Event(type=EventType.LOG,
                                 message=f"task {task.id} failed, retrying "
                                         f"(attempt {task.attempts} of {max_retries + 1})")
                        )
                    else:
                        self._say(f"[red]Task {task_id} failed: {detail}[/red]")
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    task.attempts += 1
                    task.error = f"{type(exc).__name__}: {exc}"
                    task.finished_at = datetime.now(timezone.utc).isoformat()
                    task.status = "pending" if task.attempts <= max_retries else "failed"
                    self.event_bus.emit(
                        Event(
                            type=EventType.ERROR,
                            message=f"task {task.id} raised: {task.error}",
                            data={"task": task.id, "error": task.error,
                                  "attempts": task.attempts},
                        )
                    )
                    self.event_bus.emit(
                        Event(
                            type=EventType.TASK_FINISHED,
                            data={
                                "task": task.id,
                                "success": False,
                                "summary": str(task.error or ""),
                            },
                        )
                    )
                    task_id = self._escape(task.id)
                    detail = self._escape(self._one_line(task.error))
                    if task.status == "pending":
                        self._say(f"[yellow]Task {task_id} failed: {detail} - retrying[/yellow]")
                        self.event_bus.emit(
                            Event(type=EventType.LOG,
                                 message=f"task {task.id} failed, retrying "
                                         f"(attempt {task.attempts} of {max_retries + 1})")
                        )
                    else:
                        self._say(f"[red]Task {task_id} failed: {detail}[/red]")

        async def run_tasks(tasks: list[TaskModel]) -> None:
            """Run tasks concurrently, never swallowing a cancellation.

            Exceptions raised inside `run_one` are converted into task failures
            there; only `asyncio.CancelledError` may still reach us and it must
            always propagate.
            """
            if not tasks:
                return
            results = await asyncio.gather(*[run_one(t) for t in tasks],
                                           return_exceptions=True)
            for res in results:
                if isinstance(res, asyncio.CancelledError):
                    raise res

        async def run_retries(tasks: list[TaskModel]) -> None:
            """Re-run tasks that failed while retries remain, until the budget is spent."""
            retry_queue = [t for t in tasks
                           if t.status == "pending" and t.attempts <= max_retries]
            while retry_queue:
                batch = retry_queue
                retry_queue = []
                await run_tasks(batch)
                for t in batch:
                    if t.status == "pending" and t.attempts <= max_retries:
                        retry_queue.append(t)

        for wave in waves:
            failed_ids = {t.id for t in self.tasks if t.status == "failed"}
            runnable = []
            for t in wave:
                if any(d in failed_ids for d in t.dependencies):
                    t.status = "blocked"
                else:
                    runnable.append(t)
            await run_tasks(runnable)
            await run_retries(runnable)
