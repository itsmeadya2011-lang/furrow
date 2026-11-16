"""Task execution for the Furrow orchestrator.

Handles running tasks in waves, managing retries, and handling failures.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, Usage
from furrow.core.events import Event, EventBus, EventType


class TaskExecutor:
    """Executes tasks in dependency-ordered waves with retry logic."""

    def __init__(
        self,
        settings: Settings,
        client: Any,
        workspace: Any,
        original_goal: str,
        event_bus: EventBus,
    ) -> None:
        self.settings = settings
        self.client = client
        self.workspace = workspace
        self.original_goal = original_goal
        self.event_bus = event_bus
        self._workers: dict[str, WorkerAgent] = {}

    def _worker_for(self, task: TaskModel) -> WorkerAgent:
        """Return the worker for *task*, creating it once so retries reuse it."""
        worker = self._workers.get(task.id)
        if worker is None:
            worker = WorkerAgent(
                task=task,
                client=self.client,
                settings=self.settings,
                workspace=self.workspace,
                goal=self.original_goal,
            )
            self._workers[task.id] = worker
        return worker

    async def _run_one(
        self,
        task: TaskModel,
        skip_ids: set[str],
        max_retries: int,
        _escape: Any,
        _one_line: Any,
        _say: Any,
    ) -> None:
        """Run a single task, handling success, failure, and retries."""
        if task.id in skip_ids:
            self.event_bus.emit(
                Event(type=EventType.LOG, message=f"task {task.id} already completed - skipping")
            )
            return

        sem = asyncio.Semaphore(self.settings.max_parallel_tasks)
        async with sem:
            task.status = "running"
            task.started_at = datetime.now(timezone.utc).isoformat()
            self.event_bus.emit(Event(type=EventType.TASK_STARTED, data={"task": task.id}))
            try:
                result = await self._worker_for(task).run()
            except asyncio.CancelledError:
                raise  # cancellation is not a task failure
            except Exception as exc:
                # A raised exception is handled exactly like a worker that
                # returned success=False: it consumes one attempt and follows
                # the same retry / terminal-failure path.
                await self._handle_task_error(task, exc, max_retries, _escape, _one_line, _say)
                return

            await self._handle_task_result(task, result, max_retries, _escape, _one_line, _say)

    async def _handle_task_error(
        self,
        task: TaskModel,
        exc: Exception,
        max_retries: int,
        _escape: Any,
        _one_line: Any,
        _say: Any,
    ) -> None:
        """Handle an exception raised during task execution."""
        task.attempts += 1
        task.error = f"{type(exc).__name__}: {exc}"
        task.finished_at = datetime.now(timezone.utc).isoformat()
        task.status = "pending" if task.attempts <= max_retries else "failed"
        self.event_bus.emit(
            Event(
                type=EventType.ERROR,
                message=f"task {task.id} raised: {task.error}",
                data={"task": task.id, "error": task.error, "attempts": task.attempts},
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
        task_id = _escape(task.id)
        detail = _escape(_one_line(task.error))
        if task.status == "pending":
            _say(f"[yellow]Task {task_id} failed: {detail} - retrying[/yellow]")
            self.event_bus.emit(
                Event(
                    type=EventType.LOG,
                    message=f"task {task.id} failed, retrying (attempt {task.attempts} of {max_retries + 1})",
                )
            )
        else:
            _say(f"[red]Task {task_id} failed: {detail}[/red]")

    async def _handle_task_result(
        self,
        task: TaskModel,
        result: Any,
        max_retries: int,
        _escape: Any,
        _one_line: Any,
        _say: Any,
    ) -> None:
        """Handle a WorkerResult from task execution."""
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
            task_id = _escape(task.id)
            detail = _escape(_one_line(task.error or "unknown error"))
            if task.status == "completed":
                _say(f"[green]Task {task_id} completed[/green]")
            elif task.status == "pending":
                _say(f"[yellow]Task {task_id} failed: {detail} - retrying[/yellow]")
                self.event_bus.emit(
                    Event(
                        type=EventType.LOG,
                        message=f"task {task.id} failed, retrying (attempt {task.attempts} of {max_retries + 1})",
                    )
                )
            else:
                _say(f"[red]Task {task_id} failed: {detail}[/red]")
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
                    data={"task": task.id, "error": task.error, "attempts": task.attempts},
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
            task_id = _escape(task.id)
            detail = _escape(_one_line(task.error))
            if task.status == "pending":
                _say(f"[yellow]Task {task_id} failed: {detail} - retrying[/yellow]")
                self.event_bus.emit(
                    Event(
                        type=EventType.LOG,
                        message=f"task {task.id} failed, retrying (attempt {task.attempts} of {max_retries + 1})",
                    )
                )
            else:
                _say(f"[red]Task {task_id} failed: {detail}[/red]")

    async def execute(
        self,
        tasks: list[TaskModel],
        waves: list[list[TaskModel]],
        skip_ids: set[str],
        _escape: Any,
        _one_line: Any,
        _say: Any,
    ) -> None:
        """Execute all tasks in the given waves."""
        max_retries = max(0, self.settings.max_task_retries)

        async def run_tasks(tasks: list[TaskModel]) -> None:
            """Run tasks concurrently, never swallowing a cancellation."""
            if not tasks:
                return
            results = await asyncio.gather(
                *[self._run_one(t, skip_ids, max_retries, _escape, _one_line, _say) for t in tasks],
                return_exceptions=True,
            )
            for res in results:
                if isinstance(res, asyncio.CancelledError):
                    raise res

        async def run_retries(tasks: list[TaskModel]) -> None:
            """Re-run tasks that failed while retries remain, until the budget is spent."""
            retry_queue = [t for t in tasks if t.status == "pending" and t.attempts <= max_retries]
            while retry_queue:
                batch = retry_queue
                retry_queue = []
                await run_tasks(batch)
                for t in batch:
                    if t.status == "pending" and t.attempts <= max_retries:
                        retry_queue.append(t)

        # Outcomes of THIS execution only. A task that fails or is blocked here
        # blocks its dependents; a stale failed task from an earlier cycle (one
        # that is not part of the current plan) must not block a fresh task.
        unavailable: set[str] = set()
        for wave in waves:
            runnable = []
            for t in wave:
                if t.status == "failed":
                    # The retry budget was exhausted in an earlier cycle and the
                    # plan kept the task unchanged: it stays failed and is not
                    # executed again.
                    self.event_bus.emit(
                        Event(type=EventType.LOG, message=f"task {t.id} already failed - not retrying")
                    )
                    unavailable.add(t.id)
                    continue
                if any(d in unavailable for d in t.dependencies):
                    t.status = "blocked"
                    unavailable.add(t.id)
                    continue
                runnable.append(t)
            await run_tasks(runnable)
            await run_retries(runnable)
            for t in wave:
                if t.status in {"failed", "blocked"}:
                    unavailable.add(t.id)