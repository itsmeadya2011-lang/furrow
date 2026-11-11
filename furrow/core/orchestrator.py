from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, RunResult, Settings, TaskModel, Usage, WorkerResult
from furrow.config import settings as global_settings
from furrow.core.events import Event, EventBus, EventType
from furrow.core.state import HistoryEntry, RunState, StateStore
from furrow.llm import LLMClient
from furrow.logging import get_logger

_UNSET = object()

_MAX_TEXT = 120

logger = get_logger(__name__)


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
            logger.info("cycle_started", cycle=self.cycles, goal=self.goal[:200])
            self._say(f"[bold]═══ Cycle {self.cycles} ═══[/bold]")
            self.event_bus.emit(Event(type=EventType.CYCLE_STARTED, data={"cycle": self.cycles}))
            try:
                # Collect files changed in previous cycles
                all_files_changed = sorted(
                    {f for t in self.tasks for f in (getattr(t, "files_changed", None) or [])}
                )
                plan = await self.planner.plan(
                    self.goal,
                    context=self._context(),
                    task_history=self.tasks,
                    files_changed=all_files_changed,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("planner_failed", cycle=self.cycles, error=str(e))
                self.event_bus.emit(Event(type=EventType.ERROR, message=str(e)))
                self._stopped_reason = f"planner failed: {e}"
                self.cycles -= 1
                break
            logger.info("plan_created", cycle=self.cycles, tasks=len(plan.tasks), rationale=plan.rationale[:200])
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
                logger.error("execute_failed", cycle=self.cycles, error=str(e))
                self.event_bus.emit(Event(type=EventType.ERROR, message=str(e)))
                self._stopped_reason = f"execute failed: {e}"
                self.cycles -= 1
                break
            try:
                test_result = await self.tester.run(self.original_goal, self.tasks)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("tester_failed", cycle=self.cycles, error=str(e))
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
            logger.info("test_result", cycle=self.cycles, passed=test_result.passed, skipped=skipped, summary=summary[:200])
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
            logger.info("cycle_completed", cycle=self.cycles, usage=cycle_usage.model_dump())
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
                logger.info("state_saved", cycle=self.cycles, path=str(self.state_store.path))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # The state was NOT written: report the failure (including the
                # path that could not be written) and do NOT emit STATE_SAVED,
                # which would tell consumers the state was saved when it was not.
                logger.error("state_save_failed", cycle=self.cycles, path=str(self.state_store.path), error=str(e))
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

        active = (
            [t for t in self.tasks if t.id in self._active_ids] if self._active_ids else self.tasks
        )
        completed = sum(1 for t in active if t.status == "completed")
        failed = sum(1 for t in active if t.status == "failed")
        blocked = sum(1 for t in active if t.status == "blocked")
        files_changed = sorted(
            {f for t in active for f in (getattr(t, "files_changed", None) or [])}
        )

        logger.info("run_completed", cycles=self.cycles, stopped_reason=self._stopped_reason,
                   tasks_completed=completed, tasks_failed=failed, tasks_blocked=blocked)
        self._say(f"Stopped: {self._escape(self._one_line(self._stopped_reason))}")

        result = RunResult(
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

        # Send webhook notification if configured
        await self._send_webhook(result)

        return result

    async def _send_webhook(self, result: RunResult) -> None:
        """Send run completion webhook if configured."""
        webhook_url = getattr(self.settings, "webhook_url", None)
        if not webhook_url:
            return
        try:
            import httpx
            payload = {
                "goal": self.original_goal,
                "cycles": result.cycles,
                "passed": result.passed,
                "stopped_reason": result.stopped_reason,
                "tasks_completed": result.tasks_completed,
                "tasks_failed": result.tasks_failed,
                "tasks_blocked": result.tasks_blocked,
                "tests_run": result.tests_run,
                "usage": result.usage.model_dump() if result.usage else {},
                "files_changed": result.files_changed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(webhook_url, json=payload)
            logger.info("webhook_sent", url=webhook_url)
        except Exception as exc:
            logger.warning("webhook_failed", url=webhook_url, error=str(exc))

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
                    if old.attempts <= max_retries and (
                        old.status == "failed" or old.status in ("pending", "running")
                    ):
                        pt.status = "pending"
                    else:
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
        """Execute all tasks in the plan, respecting dependencies and retries."""
        waves = self._plan_waves(plan.tasks)
        self._semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        self._max_retries = max(0, self.settings.max_task_retries)
        self._workers: dict[str, WorkerAgent] = {}

        for wave in waves:
            await self._execute_wave(wave)

    def _worker_for(self, task: TaskModel) -> WorkerAgent:
        """Return the worker for *task*, creating it once so retries reuse it."""
        worker = self._workers.get(task.id)
        if worker is None:
            worker = WorkerAgent(task=task, client=self.client, settings=self.settings,
                                 workspace=self.settings.workspace, goal=self.original_goal)
            self._workers[task.id] = worker
        return worker

    async def _execute_wave(self, wave: list[TaskModel]) -> None:
        """Execute a single wave of tasks, then handle retries."""
        failed_ids = {t.id for t in self.tasks if t.status == "failed"}
        runnable = self._filter_runnable(wave, failed_ids)
        await self._run_tasks(runnable)
        await self._run_retries(runnable)

    def _filter_runnable(self, wave: list[TaskModel], failed_ids: set[str]) -> list[TaskModel]:
        """Return tasks from *wave* whose dependencies are not failed."""
        runnable = []
        for t in wave:
            if any(d in failed_ids for d in t.dependencies):
                t.status = "blocked"
            else:
                runnable.append(t)
        return runnable

    async def _run_tasks(self, tasks: list[TaskModel]) -> None:
        """Run tasks concurrently, never swallowing a cancellation."""
        if not tasks:
            return
        results = await asyncio.gather(*[self._run_one(t) for t in tasks],
                                       return_exceptions=True)
        for res in results:
            if isinstance(res, asyncio.CancelledError):
                raise res

    async def _run_retries(self, tasks: list[TaskModel]) -> None:
        """Re-run tasks that failed while retries remain, until the budget is spent."""
        retry_queue = [t for t in tasks
                       if t.status == "pending" and t.attempts <= self._max_retries]
        while retry_queue:
            batch = retry_queue
            retry_queue = []
            await self._run_tasks(batch)
            for t in batch:
                if t.status == "pending" and t.attempts <= self._max_retries:
                    retry_queue.append(t)

    async def _run_one(self, task: TaskModel) -> None:
        """Execute a single task with error handling and retry logic."""
        if task.id in self._skip_ids:
            self.event_bus.emit(Event(type=EventType.LOG,
                                    message=f"task {task.id} already completed - skipping"))
            return

        # Capture client usage before task to calculate per-task cost
        usage_before = self._client_usage()

        async with self._semaphore:
            task.status = "running"
            task.started_at = datetime.now(timezone.utc).isoformat()
            self.event_bus.emit(Event(type=EventType.TASK_STARTED, data={"task": task.id}))

            # Execute the worker
            result = await self._execute_worker(task)

            # If the worker raised an exception, it was already handled in _execute_worker
            # (which emits events and updates task status). We just need to check if
            # the task is already in a terminal state.
            if result is None:
                return

            if task.status in ("completed", "failed", "blocked"):
                return

            # Process the result
            await self._process_worker_result(task, result)

        # Calculate per-task cost
        usage_after = self._client_usage()
        task_usage = Usage(
            input_tokens=max(0, usage_after.input_tokens - usage_before.input_tokens),
            output_tokens=max(0, usage_after.output_tokens - usage_before.output_tokens),
            calls=max(0, usage_after.calls - usage_before.calls),
            cost_usd=max(0.0, usage_after.cost_usd - usage_before.cost_usd),
        )
        task.cost_usd = task_usage.cost_usd

    async def _execute_worker(self, task: TaskModel) -> WorkerResult | None:
        """Execute the worker for a task, handling exceptions.

        Returns the worker's result on success, or None when an exception was
        raised and already fully handled (attempts incremented, status set,
        events emitted) - in which case the caller must not re-process the
        result.
        """
        usage_before = self._client_usage()
        try:
            return await self._worker_for(task).run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Convert exception into a failed WorkerResult so retry logic is uniform
            # Also emit ERROR event to match original behavior
            error_msg = f"{type(exc).__name__}: {exc}"
            task.attempts += 1
            task.error = error_msg
            task.finished_at = datetime.now(timezone.utc).isoformat()
            task.status = "pending" if task.attempts <= self._max_retries else "failed"
            self.event_bus.emit(
                Event(
                    type=EventType.ERROR,
                    message=f"task {task.id} raised: {error_msg}",
                    data={"task": task.id, "error": error_msg, "attempts": task.attempts},
                )
            )
            self.event_bus.emit(
                Event(
                    type=EventType.TASK_FINISHED,
                    data={
                        "task": task.id,
                        "success": False,
                        "summary": str(error_msg),
                    },
                )
            )
            task_id = self._escape(task.id)
            detail = self._escape(self._one_line(error_msg))
            if task.status == "pending":
                self._say(f"[yellow]Task {task_id} failed: {detail} - retrying[/yellow]")
                self.event_bus.emit(
                    Event(type=EventType.LOG,
                         message=f"task {task.id} failed, retrying "
                                 f"(attempt {task.attempts} of {self._max_retries + 1})")
                )
            else:
                self._say(f"[red]Task {task_id} failed: {detail}[/red]")
            # Capture cost for failed task
            usage_after = self._client_usage()
            task_usage = Usage(
                input_tokens=max(0, usage_after.input_tokens - usage_before.input_tokens),
                output_tokens=max(0, usage_after.output_tokens - usage_before.output_tokens),
                calls=max(0, usage_after.calls - usage_before.calls),
                cost_usd=max(0.0, usage_after.cost_usd - usage_before.cost_usd),
            )
            task.cost_usd = task_usage.cost_usd
            return None

    async def _process_worker_result(self, task: TaskModel, result: WorkerResult) -> None:
        """Update task state based on worker result and emit events."""
        try:
            success = bool(getattr(result, "success", False))
        except Exception as exc:
            # Attribute access failed (e.g., property raises) - treat as failure
            error_msg = f"{type(exc).__name__}: {exc}"
            task.error = error_msg
            task.attempts += 1
            task.finished_at = datetime.now(timezone.utc).isoformat()
            task.status = "pending" if task.attempts <= self._max_retries else "failed"
            self.event_bus.emit(
                Event(
                    type=EventType.ERROR,
                    message=f"task {task.id} raised: {error_msg}",
                    data={"task": task.id, "error": error_msg, "attempts": task.attempts},
                )
            )
            self.event_bus.emit(
                Event(
                    type=EventType.TASK_FINISHED,
                    data={
                        "task": task.id,
                        "success": False,
                        "summary": str(error_msg),
                    },
                )
            )
            task_id = self._escape(task.id)
            detail = self._escape(self._one_line(error_msg))
            if task.status == "pending":
                self._say(f"[yellow]Task {task_id} failed: {detail} - retrying[/yellow]")
                self.event_bus.emit(
                    Event(type=EventType.LOG,
                         message=f"task {task.id} failed, retrying "
                                 f"(attempt {task.attempts} of {self._max_retries + 1})")
                )
            else:
                self._say(f"[red]Task {task_id} failed: {detail}[/red]")
            return

        task.result = getattr(result, "summary", None)
        task.error = getattr(result, "error", None)
        task.attempts += 1
        task.files_changed = sorted(
            {str(p) for p in (getattr(result, "files_changed", None) or [])}
        )
        task.finished_at = datetime.now(timezone.utc).isoformat()

        if success:
            task.status = "completed"
        elif task.attempts <= self._max_retries:
            task.status = "pending"
        else:
            task.status = "failed"

        self._emit_task_finished_event(task)
        self._log_task_completion(task)

    def _emit_task_finished_event(self, task: TaskModel) -> None:
        """Emit TASK_FINISHED event for a completed task."""
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

    def _log_task_completion(self, task: TaskModel) -> None:
        """Log task completion status to console."""
        task_id = self._escape(task.id)
        detail = self._escape(self._one_line(task.error or "unknown error"))
        if task.status == "completed":
            self._say(f"[green]Task {task_id} completed[/green]")
        elif task.status == "pending":
            self._say(f"[yellow]Task {task_id} failed: {detail} - retrying[/yellow]")
            self.event_bus.emit(
                Event(type=EventType.LOG,
                     message=f"task {task.id} failed, retrying "
                             f"(attempt {task.attempts} of {self._max_retries + 1})")
            )
        else:
            self._say(f"[red]Task {task_id} failed: {detail}[/red]")
