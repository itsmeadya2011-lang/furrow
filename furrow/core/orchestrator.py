from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, RunResult, Settings, TaskModel
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
        self._console_obj: Any = _UNSET
        if resume:
            st = self.state_store.load()
            if st is not None:
                self.goal = st.goal
                self.original_goal = st.original_goal
                self.cycles = st.cycles
                self.tasks = list(st.tasks)
                self.history = list(st.history)

    def status(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "original_goal": self.original_goal,
            "cycles": self.cycles,
            "stopped_reason": self._stopped_reason,
            "tasks": [t.model_dump() for t in self.tasks],
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
        if self._resume:
            st = self.state_store.load()
            if st is None:
                self.event_bus.emit(Event(type=EventType.LOG,
                                        message="no state file found, starting fresh"))
            else:
                self.goal = st.goal
                self.original_goal = st.original_goal
                self.cycles = st.cycles
                self.tasks = list(st.tasks)
                self.history = list(st.history)

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
            except ValueError as e:
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
            await self._execute(plan)
            self.event_bus.emit(Event(type=EventType.TEST_STARTED, data={}))
            test_result = await self.tester.run(self.original_goal, self.tasks)
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
            if not test_result.passed:
                self._feedback = "\n".join(test_result.failures)
                self.goal = (f"{self.original_goal}\n\n"
                             "The previous cycle left failing tests. Fix them:\n"
                             f"{self._feedback}")
            self.history.append(HistoryEntry(
                cycle=self.cycles, goal=self.goal, plan_rationale=plan.rationale,
                task_summaries=[t.id for t in self.tasks],
                tests_passed=test_result.passed, test_summary=test_result.summary,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
            self.event_bus.emit(Event(type=EventType.CYCLE_FINISHED, data={"cycle": self.cycles}))
            state = RunState(goal=self.goal, original_goal=self.original_goal,
                            cycles=self.cycles, tasks=self.tasks, history=self.history)
            self.state_store.save(state)
            self.event_bus.emit(Event(type=EventType.STATE_SAVED,
                                    data={"path": str(self.state_store.path)}))
            sig = "|".join(f"{t.id}:{t.description}" for t in plan.tasks)
            if (not test_result.passed) and sig == self._prev_sig:
                self._no_progress += 1
                if self._no_progress >= 2:
                    self._stopped_reason = "no progress"
                    break
            else:
                if sig != self._prev_sig or test_result.passed:
                    self._no_progress = 0
                self._prev_sig = sig

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
        return RunResult(
            passed=bool(self._last_test_passed),
            cycles=self.cycles,
            tasks_completed=completed,
            tasks_failed=failed,
            summary=f"stopped: {self._stopped_reason}",
            stopped_reason=self._stopped_reason,
        )

    def _context(self) -> str:
        if self._feedback:
            return self._feedback
        return ""

    def _merge_plan(self, plan: Plan) -> None:
        by_id = {t.id: t for t in self.tasks}
        for index, pt in enumerate(plan.tasks):
            old = by_id.get(pt.id)
            if old is not None:
                if old.status == "completed":
                    pt.status = "completed"
                else:
                    pt.status = "pending"
                old.description = pt.description
                old.files = pt.files
                old.dependencies = pt.dependencies
                old.status = pt.status
                old.result = pt.result
                old.error = pt.error
                if pt.status != "completed":
                    old.attempts = 0
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
            if task.status == "completed":
                return
            async with sem:
                task.status = "running"
                task.started_at = datetime.now(timezone.utc).isoformat()
                self.event_bus.emit(Event(type=EventType.TASK_STARTED, data={"task": task.id}))
                result = await worker_for(task).run()
                if isinstance(result, BaseException):
                    success = False
                    summary = str(result)
                    error = str(result)
                else:
                    success = bool(getattr(result, "success", False))
                    summary = getattr(result, "summary", None)
                    error = getattr(result, "error", None)
                task.result = summary
                task.error = error
                task.attempts += 1
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

        async def run_retries(tasks: list[TaskModel]) -> None:
            """Re-run tasks that failed while retries remain, until the budget is spent."""
            retry_queue = [t for t in tasks
                           if t.status == "pending" and t.attempts <= max_retries]
            while retry_queue:
                batch = retry_queue
                retry_queue = []
                await asyncio.gather(*[run_one(t) for t in batch], return_exceptions=True)
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
            await asyncio.gather(*[run_one(t) for t in runnable], return_exceptions=True)
            await run_retries(runnable)
