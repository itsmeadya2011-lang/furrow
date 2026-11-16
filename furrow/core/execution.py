"""Task execution functionality for the orchestrator."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from furrow.config import Settings, TaskModel
from furrow.core.events import Event, EventBus, EventType


async def execute_plan(
    plan_waves: list[list[TaskModel]],
    settings: Settings,
    client: Any,
    workspace: Any,
    goal: str,
    skip_ids: set[str],
    event_bus: EventBus,
    tasks: list[TaskModel],
    worker_cls: Callable[..., Any] | None = None,
    say: Callable[[str], None] | None = None,
) -> None:
    """Execute a plan by running tasks in waves with retry logic."""
    if worker_cls is None:
        from furrow.agents.worker import WorkerAgent

        worker_cls = WorkerAgent
    if say is None:
        def say(message: str) -> None:  # pragma: no cover - default no-op
            pass
    sem = asyncio.Semaphore(settings.max_parallel_tasks)
    max_retries = max(0, settings.max_task_retries)
    workers: dict[str, Any] = {}

    def worker_for(task: TaskModel) -> Any:
        """Return the worker for *task*, creating it once so retries reuse it."""
        worker = workers.get(task.id)
        if worker is None:
            worker = worker_cls(
                task=task,
                client=client,
                settings=settings,
                workspace=workspace,
                goal=goal,
            )
            workers[task.id] = worker
        return worker

    def _clip(text: str, limit: int = 120) -> str:
        flat = " ".join(str(text).split())
        if len(flat) <= limit:
            return flat
        return flat[: limit - 1].rstrip() + "…"

    def _report(task: TaskModel, detail: str) -> None:
        """Print a one-line progress message for *task*; never raises."""
        try:
            task_id = _clip(task.id)
            msg_detail = _clip(detail)
            if task.status == "completed":
                say(f"[green]Task {task_id} completed[/green]")
            elif task.status == "pending":
                say(
                    f"[yellow]Task {task_id} failed: {msg_detail} - retrying[/yellow]"
                )
                event_bus.emit(
                    Event(
                        type=EventType.LOG,
                        message=(
                            f"task {task.id} failed, retrying "
                            f"(attempt {task.attempts} of {max_retries + 1})"
                        ),
                    )
                )
            else:
                say(f"[red]Task {task_id} failed: {msg_detail}[/red]")
        except Exception:  # pragma: no cover - reporting must never break a run
            pass

    async def run_one(task: TaskModel) -> None:
        if task.id in skip_ids:
            event_bus.emit(
                Event(
                    type=EventType.LOG,
                    message=f"task {task.id} already completed - skipping",
                )
            )
            return
        async with sem:
            task.status = "running"
            task.started_at = datetime.now(timezone.utc).isoformat()
            event_bus.emit(Event(type=EventType.TASK_STARTED, data={"task": task.id}))
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
                task.status = (
                    "pending" if task.attempts <= max_retries else "failed"
                )
                event_bus.emit(
                    Event(
                        type=EventType.ERROR,
                        message=f"task {task.id} raised: {task.error}",
                        data={
                            "task": task.id,
                            "error": task.error,
                            "attempts": task.attempts,
                        },
                    )
                )
                event_bus.emit(
                    Event(
                        type=EventType.TASK_FINISHED,
                        data={
                            "task": task.id,
                            "success": False,
                            "summary": str(task.error or ""),
                        },
                    )
                )
                _report(task, task.error)
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
                event_bus.emit(
                    Event(
                        type=EventType.TASK_FINISHED,
                        data={
                            "task": task.id,
                            "success": task.status == "completed",
                            "summary": str(task.result or ""),
                        },
                    )
                )
                _report(task, task.error or "unknown error")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                task.attempts += 1
                task.error = f"{type(exc).__name__}: {exc}"
                task.finished_at = datetime.now(timezone.utc).isoformat()
                task.status = (
                    "pending" if task.attempts <= max_retries else "failed"
                )
                event_bus.emit(
                    Event(
                        type=EventType.ERROR,
                        message=f"task {task.id} raised: {task.error}",
                        data={
                            "task": task.id,
                            "error": task.error,
                            "attempts": task.attempts,
                        },
                    )
                )
                event_bus.emit(
                    Event(
                        type=EventType.TASK_FINISHED,
                        data={
                            "task": task.id,
                            "success": False,
                            "summary": str(task.error or ""),
                        },
                    )
                )
                _report(task, task.error)

    async def run_tasks(tasks_to_run: list[TaskModel]) -> None:
        """Run tasks concurrently, never swallowing a cancellation."""
        if not tasks_to_run:
            return
        results = await asyncio.gather(
            *[run_one(t) for t in tasks_to_run], return_exceptions=True
        )
        for res in results:
            if isinstance(res, asyncio.CancelledError):
                raise res

    async def run_retries(tasks_to_retry: list[TaskModel]) -> None:
        """Re-run tasks that failed while retries remain, until the budget is spent."""
        retry_queue = [
            t
            for t in tasks_to_retry
            if t.status == "pending" and t.attempts <= max_retries
        ]
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
    for wave in plan_waves:
        runnable = []
        for t in wave:
            if t.status == "failed":
                # The retry budget was exhausted in an earlier cycle and the
                # plan kept the task unchanged: it stays failed and is not
                # executed again.
                event_bus.emit(
                    Event(
                        type=EventType.LOG,
                        message=f"task {t.id} already failed - not retrying",
                    )
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