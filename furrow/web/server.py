"""HTTP and WebSocket front end for Furrow.

The page served at ``/`` starts one autonomous run per WebSocket connection and
streams the orchestrator's progress back as JSON frames.

WebSocket protocol (``/ws``, one run per connection)
----------------------------------------------------

The server accepts the socket and then reads **exactly one** ``start`` message.
Everything the server sends is a JSON object with a string ``event`` and a
string ``message``; ``data`` is an object on the ``accepted`` and ``done``
frames and absent everywhere else.  Frames are sent as they happen and the
server closes the socket (code ``1000``) once the run is over, so a client that
wants another run opens a new connection.

Client -> server, exactly once per connection::

    {"type": "start", "goal": "<non-empty string, <= 4000 chars>",
     "model": "<optional model id>", "max_cycles": "<optional int 1..50>",
     "token": "<required only when FURROW_WEB_TOKEN is set>"}

``type`` may be omitted (``{"goal": "..."}`` is accepted as a ``start``).
Client -> server while the run is in flight::

    {"type": "cancel"}

Server -> client::

    {"event": "accepted", "message": "...", "data": {"run_id", "max_cycles"}}
    {"event": "<name>", "message": "..."}     # run, plan, cycle, task, test, error
    {"event": "finished", "message": "..."}   # the loop stopped; the message says why
    {"event": "warn", "message": "..."}       # server-side event stream overflow
    {"event": "error", "message": "..."}      # rejected input or a failed run; socket closes after
    {"event": "done", "message": "...", "data": {"done", "cycles", "error"}}

Every event name carries exactly one meaning, so no frame can be confused with
another:

* ``finished`` is the orchestrator's own terminal event (goal met, cycle budget,
  stop request, agent error, cancellation) relayed under a name of its own.  The
  orchestrator calls it ``done``; the web layer renames it so that ``done``
  keeps a single, unambiguous meaning on the wire;
* ``done`` is the **summary** and is sent exactly once per connection, always
  last and always with ``data``.

A client may therefore act the moment it sees ``done`` - it cannot lose the
summary by reacting to the loop's terminal event first, which is exactly what a
client keyed on ``event == "done"`` used to do.

A second ``start`` on a live connection is refused with an ``error`` frame and
ignored; the run already in progress is not disturbed.

Safety properties
------------------

* :func:`run` binds ``127.0.0.1`` by default.  The agent writes files, so
  exposing the server is an explicit ``--host 0.0.0.0`` decision, and the CLI
  prints a warning when that is done without an API token.
* Optional, opt-in authentication: set ``FURROW_WEB_TOKEN`` and every run
  needs the matching ``token`` value (``GET /`` then needs ``?token=`` or the
  ``X-Furrow-Token`` header).  ``GET /health`` stays open for probes and never
  exposes run data.
* Web runs are always bounded: a run gets :data:`DEFAULT_WEB_MAX_CYCLES`
  cycles (or the operator's smaller ``max_cycles`` setting, or the client's own
  smaller ``max_cycles``), clamped to :data:`HARD_MAX_WEB_CYCLES`.  The CLI
  default of "unlimited" is never changed by the web layer.
* At most :data:`DEFAULT_MAX_CONCURRENT_RUNS` runs execute at once
  (``FURROW_WEB_MAX_CONCURRENCY`` or ``run(max_concurrent_runs=...)``).
* A client disconnect stops the run cooperatively via ``Orchestrator.stop()``;
  if the run has not unwound within :data:`STOP_GRACE_SECONDS` it is cancelled
  and awaited, so no task is left pending behind a dead socket.
* Inputs are validated: a non-object payload, a missing/empty goal, a goal
  longer than :data:`MAX_GOAL_LENGTH` and an unknown message type are all
  refused with an ``error`` frame.

The run summary is kept in process memory and exposed read-only at
``GET /api/state``; a multi-worker ``uvicorn`` invocation would keep one
summary per worker.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import json
import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from string import Template
from typing import Any, Literal

import structlog
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError

from furrow.config import DEFAULT_MAX_CONCURRENT_RUNS as CONFIG_MAX_CONCURRENT_RUNS
from furrow.config import Settings, configure_logging
from furrow.config import settings as global_settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

log = structlog.get_logger(__name__)

DEFAULT_HOST = "127.0.0.1"
"""Loopback by default: the agent writes files, exposure must be a choice."""

DEFAULT_PORT = 8000

DEFAULT_WEB_MAX_CYCLES = 8
"""Cycle budget of a run started from a browser.

A cycle is a full plan/execute/test round, so eight is a generous amount of
work for an unauthenticated socket while still bounding the wall-clock and API
spend of a single connection.  ``0`` (unlimited, the CLI default) would pin the
socket forever, so the web layer substitutes this value instead.
"""

HARD_MAX_WEB_CYCLES = 50
"""Upper bound for a web run, whatever the client or settings ask for."""

DEFAULT_MAX_CONCURRENT_RUNS = CONFIG_MAX_CONCURRENT_RUNS
"""Runs served in parallel; every run can spawn ``max_parallel_tasks`` workers.

Re-exported from :mod:`furrow.config` so the number has one definition. The value
actually used is :attr:`Settings.max_concurrent_runs`; see :func:`_max_concurrent_runs`.
"""

MAX_CONCURRENT_RUNS_ENV = "FURROW_WEB_MAX_CONCURRENCY"
TOKEN_ENV = "FURROW_WEB_TOKEN"
TOKEN_HEADER = "X-Furrow-Token"

MAX_GOAL_LENGTH = 4000
MAX_MODEL_LENGTH = 200
MAX_SUMMARY_TEXT = 400

EVENT_QUEUE_MAXSIZE = 512
RELAY_POLL_SECONDS = 0.1
"""How often the relay wakes to re-check the run and the stop grace period."""

STOP_GRACE_SECONDS = 10.0
"""Time a stopped or disconnected run may take to unwind before it is cancelled."""

CLOSE_CODE_NORMAL = 1000
CANCELLED_REASON = "the run was cancelled"

FINISHED_EVENT = "finished"
"""Wire name of the loop's own terminal event; ``done`` is reserved for the summary.

The name is deliberately neutral: the loop emits this event whether it met the
goal or not, and ``message`` carries the reason.  ``halt`` would have been a
lie on the happy path, where the message is "Goal complete.".
"""

_WIRE_EVENT_NAMES = {"done": FINISHED_EVENT}
"""Loop events renamed on the wire, so that no event name carries two meanings."""

app = FastAPI(title="Furrow")


# --------------------------------------------------------------------- wire schema


class StartRequest(BaseModel):
    """The single ``start`` message a client may send on a fresh connection."""

    type: Literal["start"] = "start"
    goal: str = Field(min_length=1, max_length=MAX_GOAL_LENGTH)
    model: str | None = Field(default=None, min_length=1, max_length=MAX_MODEL_LENGTH)
    max_cycles: int | None = Field(default=None, ge=1, le=HARD_MAX_WEB_CYCLES)
    token: str | None = None


class _Rejected(Exception):
    """The payload was unusable; report it on the socket and close."""


class _ClientGone(Exception):
    """The socket closed before a usable ``start`` message arrived."""


def _describe_validation_error(exc: ValidationError) -> str:
    """Flatten a pydantic error into one readable line."""
    parts: list[str] = []
    for item in exc.errors():
        location = ".".join(str(piece) for piece in item.get("loc", ())) or "payload"
        parts.append(f"{location}: {item.get('msg', 'invalid value')}")
    return "; ".join(parts) or "invalid payload"


# ------------------------------------------------------------------- process state


_state_lock = threading.Lock()
_active_runs = 0
_last_run: _RunState | None = None

_web_settings: Settings = global_settings
_web_max_concurrent_runs: int | None = None


@dataclass
class _RunState:
    """Read-only snapshot of one run, published through ``GET /api/state``."""

    run_id: str
    goal: str
    max_cycles: int
    model: str
    started_at: str
    finished_at: str | None = None
    active: bool = True
    outcome: str = "running"
    cycles: int = 0
    done: bool = False
    error: str | None = None
    events: int = 0
    dropped_events: int = 0
    current_plan: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    last_test_result: dict[str, Any] | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "model": self.model,
            "max_cycles": self.max_cycles,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "active": self.active,
            "outcome": self.outcome,
            "cycles": self.cycles,
            "done": self.done,
            "error": self.error,
            "events": self.events,
            "dropped_events": self.dropped_events,
            "current_plan": self.current_plan,
            "history": self.history,
            "last_test_result": self.last_test_result,
        }


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def app_version() -> str:
    """Installed version of the ``furrow`` package, or ``unknown``."""
    try:
        return importlib.metadata.version("furrow")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _configured_token() -> str:
    """The API token, or an empty string when authentication is switched off."""
    return (os.environ.get(TOKEN_ENV) or "").strip()


def web_token_configured() -> bool:
    """Whether ``FURROW_WEB_TOKEN`` is set, i.e. whether runs need a token."""
    return bool(_configured_token())


def _token_ok(candidate: object) -> bool:
    """Constant-time token check; always true when no token is configured."""
    expected = _configured_token()
    if not expected:
        return True
    if not isinstance(candidate, str):
        return False
    return secrets.compare_digest(candidate.strip(), expected)


def _max_concurrent_runs() -> int:
    """Configured run slots: explicit override, ``FURROW_WEB_MAX_CONCURRENCY``, then Settings.

    ``Settings.max_concurrent_runs`` is the fallback, so ``FURROW_MAX_CONCURRENT_RUNS``
    and the ``.env`` file configure the web layer like every other setting.
    """
    if _web_max_concurrent_runs is not None:
        return _web_max_concurrent_runs
    raw = (os.environ.get(MAX_CONCURRENT_RUNS_ENV) or "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            log.warning("invalid_max_concurrent_runs", value=raw)
        else:
            if value >= 1:
                return value
            log.warning("invalid_max_concurrent_runs", value=raw)
    return int(_web_settings.max_concurrent_runs)


def _active_run_count() -> int:
    with _state_lock:
        return _active_runs


def _acquire_run_slot() -> bool:
    """Reserve one of the run slots; false when the server is already busy."""
    global _active_runs
    with _state_lock:
        if _active_runs >= _max_concurrent_runs():
            return False
        _active_runs += 1
        return True


def _release_run_slot() -> None:
    global _active_runs
    with _state_lock:
        _active_runs = max(0, _active_runs - 1)


def _web_cycle_budget() -> int:
    """Cycle budget of a web run when the client does not ask for one.

    ``FURROW_MAX_CYCLES``/``run(settings=...)`` win when they are finite, so an
    operator can tighten the bound; otherwise :data:`DEFAULT_WEB_MAX_CYCLES`
    replaces the unlimited (``0``) default of the CLI.
    """
    configured = int(_web_settings.max_cycles or 0)
    budget = configured if configured > 0 else DEFAULT_WEB_MAX_CYCLES
    if budget > HARD_MAX_WEB_CYCLES:
        budget = HARD_MAX_WEB_CYCLES
        log.warning("web_cycle_budget_clamped", budget=budget, limit=HARD_MAX_WEB_CYCLES)
    return budget


def _clip(text: object, limit: int = MAX_SUMMARY_TEXT) -> str | None:
    """Collapse *text* to a bounded single line for the state endpoint."""
    if text is None:
        return None
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


def _plan_json(plan: Any) -> dict[str, Any]:
    tasks = getattr(plan, "tasks", None) or []
    return {
        "rationale": _clip(getattr(plan, "rationale", "")),
        "tasks": [
            {
                "id": _clip(getattr(task, "id", ""), 80),
                "description": _clip(getattr(task, "description", "")),
                "status": str(getattr(task, "status", "")),
                "result": _clip(getattr(task, "result", None)),
            }
            for task in tasks
        ],
    }


def _test_result_json(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "passed": bool(getattr(result, "passed", False)),
        "summary": _clip(getattr(result, "summary", "")),
        "failures": [_clip(item) for item in (getattr(result, "failures", None) or [])],
    }


# --------------------------------------------------------------------- the session


class _WebRun:
    """One WebSocket connection and the single run it is allowed to start.

    The orchestrator runs as a background task and pushes events into a bounded
    queue; :meth:`_pump` is the only place that turns queue entries into frames,
    so ordering is preserved and a slow client can never block the loop.  A
    second task, :meth:`_watch_client`, is the single reader of client messages,
    which is what makes both ``cancel`` and disconnect detection observable
    while the run is in flight.
    """

    def __init__(self, websocket: WebSocket, slot_taken: bool = False) -> None:
        self._ws = websocket
        self._queue: asyncio.Queue[tuple[str, str, dict[str, Any] | None]] = asyncio.Queue(
            maxsize=EVENT_QUEUE_MAXSIZE
        )
        self._send_lock = asyncio.Lock()
        self._client: LLMClient | None = None
        self._orchestrator: Orchestrator | None = None
        self._run_task: asyncio.Task[bool] | None = None
        self._stop_task: asyncio.Task[None] | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._state: _RunState | None = None
        self._settings: Settings | None = None
        self._slot_taken = slot_taken
        self._client_gone = False
        self._closed = False
        self._aborted = False
        self._stop_at: float | None = None
        self._dropped = 0

    # ------------------------------------------------------------------ driving

    async def serve(self) -> None:
        """Run one validated start message to completion, or fail cleanly."""
        try:
            request = await self._receive_start()
        except _ClientGone:
            await self._close()
            return
        except _Rejected as rejection:
            log.info("web_start_rejected", reason=str(rejection))
            await self._send("error", str(rejection))
            await self._close()
            return

        try:
            self._start(request)
            self._watch_task = asyncio.create_task(
                self._watch_client(), name="furrow-ws-client-reader"
            )
            await self._pump()
        except WebSocketDisconnect:  # pragma: no cover - handled in the reader task
            self._request_stop("the client closed the connection")
        except asyncio.CancelledError:
            self._request_stop("the server is shutting down")
            raise
        except Exception as exc:
            log.warning("web_run_failed", run_id=self._run_id, error=str(exc), exc_info=True)
            self._aborted = True
            await self._send("error", f"{type(exc).__name__}: {_clip(exc)}")
        finally:
            await self._finish()

    @property
    def _run_id(self) -> str:
        return self._state.run_id if self._state is not None else "-"

    async def _receive_start(self) -> StartRequest:
        """Read and validate the one ``start`` message this connection may send."""
        try:
            message = await self._ws.receive()
        except WebSocketDisconnect as exc:  # pragma: no cover - starlette version dependent
            raise _ClientGone("the client closed the connection") from exc
        if message.get("type") == "websocket.disconnect":
            raise _ClientGone("the client closed the connection")
        if message.get("bytes") is not None:
            raise _Rejected("binary frames are not supported; send a JSON object as text")
        raw_text = message.get("text") or ""
        try:
            data = json.loads(raw_text)
        except ValueError as exc:
            raise _Rejected(f"payload is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise _Rejected(f"payload must be a JSON object, got {type(data).__name__}")
        kind = data.get("type", "start")
        if kind != "start":
            raise _Rejected(f'unsupported message type {kind!r}; send {{"type": "start", ...}}')
        if not _token_ok(data.get("token")):
            raise _Rejected(f"unauthorized: {TOKEN_ENV} is set, send the matching token")
        raw_goal = data.get("goal")
        if not isinstance(raw_goal, str) or not raw_goal.strip():
            raise _Rejected('a non-empty "goal" string is required')
        goal = raw_goal.strip()
        if len(goal) > MAX_GOAL_LENGTH:
            raise _Rejected(
                f"goal is too long: {len(goal)} characters, the limit is {MAX_GOAL_LENGTH}"
            )
        payload: dict[str, Any] = {"goal": goal}
        for key in ("model", "max_cycles"):
            if data.get(key) is not None:
                payload[key] = data[key]
        try:
            return StartRequest.model_validate(payload)
        except ValidationError as exc:
            raise _Rejected(_describe_validation_error(exc)) from exc

    def _start(self, request: StartRequest) -> None:
        """Create the settings, client and orchestrator for this run."""
        budget = _web_cycle_budget()
        if request.max_cycles is not None:
            budget = min(request.max_cycles, budget, HARD_MAX_WEB_CYCLES)
        updates: dict[str, Any] = {"max_cycles": max(1, budget)}
        if request.model:
            updates.update(
                model=request.model,
                planner_model=request.model,
                worker_model=request.model,
                tester_model=request.model,
            )
        self._settings = _web_settings.model_copy(update=updates)
        self._client = LLMClient(settings=self._settings)
        self._state = _RunState(
            run_id=uuid.uuid4().hex[:12],
            goal=request.goal,
            max_cycles=int(updates["max_cycles"]),
            model=str(updates.get("model", self._settings.model)),
            started_at=_utcnow(),
        )
        self._orchestrator = Orchestrator(
            goal=request.goal,
            client=self._client,
            settings=self._settings,
            on_event=self.publish,
        )
        log.info(
            "web_run_started",
            run_id=self._state.run_id,
            max_cycles=self._state.max_cycles,
            model=self._state.model,
        )
        self._publish_accepted()
        self._run_task = asyncio.create_task(
            self._execute(), name=f"furrow-run-{self._state.run_id}"
        )

    def _publish_accepted(self) -> None:
        state = self._state
        assert state is not None
        self._queue_frame(
            "accepted",
            f"Run {state.run_id} accepted; at most {state.max_cycles} cycle(s).",
            data={"run_id": state.run_id, "max_cycles": state.max_cycles},
        )

    async def _execute(self) -> bool:
        """Body of the run task: the orchestrator plus live state collection."""
        orchestrator = self._orchestrator
        assert orchestrator is not None
        try:
            return await orchestrator.run()
        finally:
            state = self._state
            if state is not None:
                state.cycles = orchestrator.cycles
                state.current_plan = _plan_json(orchestrator.current_plan)
                state.last_test_result = _test_result_json(orchestrator.last_test_result)

    def publish(self, event: str, message: str) -> None:
        """``on_event`` sink of the orchestrator: queue a frame, never block.

        The callback is synchronous and must return immediately; the relay
        task does the awaiting.  A full queue drops the oldest frame rather
        than stalling the agent, and the relay reports the loss.

        The loop's terminal event is renamed through :data:`_WIRE_EVENT_NAMES`.
        The orchestrator calls it ``done``, which on this socket belongs to the
        session summary sent once by :meth:`_finish`; relaying it verbatim would
        put two different frames under one name, and a client acting on the
        first - as any client keyed on ``event == "done"`` does - would close
        the socket and never see the summary payload it is documented to carry.
        """
        state = self._state
        if state is not None:
            state.events += 1
        self._queue_frame(_WIRE_EVENT_NAMES.get(event, event), message)

    def _queue_frame(self, event: str, message: str, data: dict[str, Any] | None = None) -> None:
        try:
            self._queue.put_nowait((event, message, data))
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
            self._dropped += 1
            try:
                self._queue.put_nowait((event, message, data))
            except asyncio.QueueFull:  # pragma: no cover - the slot was just freed
                self._dropped += 1

    # ------------------------------------------------------------------- relaying

    async def _pump(self) -> None:
        """Send queued frames until the run task is done, then flush the rest."""
        getter: asyncio.Future[tuple[str, str, dict[str, Any] | None]] = asyncio.ensure_future(
            self._queue.get()
        )
        try:
            while True:
                if self._dropped:
                    dropped, self._dropped = self._dropped, 0
                    if self._state is not None:
                        self._state.dropped_events += dropped
                    await self._send(
                        "warn",
                        f"{dropped} event(s) dropped: the client is not reading fast enough.",
                    )
                if self._run_task is None:
                    break
                if self._run_task.done() and self._queue.empty():
                    break
                self._expire_stop()
                done, _pending = await asyncio.wait(
                    {getter}, timeout=RELAY_POLL_SECONDS, return_when=asyncio.FIRST_COMPLETED
                )
                if getter in done:
                    event, message, data = getter.result()
                    getter = asyncio.ensure_future(self._queue.get())
                    await self._send(event, message, data=data)
        finally:
            if not getter.done():
                getter.cancel()
            if getter.done() and not getter.cancelled() and getter.exception() is None:
                with contextlib.suppress(Exception):
                    event, message, data = getter.result()
                    await self._send(event, message, data=data)
            while not self._queue.empty():
                with contextlib.suppress(asyncio.QueueEmpty):
                    event, message, data = self._queue.get_nowait()
                    await self._send(event, message, data=data)

    async def _send(self, event: str, message: str, data: dict[str, Any] | None = None) -> None:
        """Best-effort frame write; a dead socket stops the run instead."""
        if self._client_gone or self._closed:
            return
        frame: dict[str, Any] = {"event": event, "message": message}
        if data is not None:
            frame["data"] = data
        async with self._send_lock:
            try:
                await self._ws.send_json(frame)
            except Exception as exc:  # noqa: BLE001 - a dead socket must not break a run
                self._client_gone = True
                log.debug("websocket_send_failed", run_id=self._run_id, error=str(exc))
                self._request_stop("the client stopped reading events")

    async def _watch_client(self) -> None:
        """Single reader of client messages: ``cancel`` and disconnect detection."""
        while True:
            try:
                message = await self._ws.receive()
            except WebSocketDisconnect:
                self._client_gone = True
                self._request_stop("the client closed the connection")
                return
            if message.get("type") == "websocket.disconnect":
                self._client_gone = True
                self._request_stop("the client closed the connection")
                return
            text = message.get("text")
            if not text:
                continue
            try:
                data = json.loads(text)
            except ValueError:
                await self._send("error", "payload is not valid JSON")
                continue
            if not isinstance(data, dict):
                await self._send("error", "payload must be a JSON object")
                continue
            kind = data.get("type")
            if kind == "cancel":
                self._request_stop("the client asked to cancel")
            elif kind == "start":
                await self._send(
                    "error",
                    "a run is already in progress on this connection; "
                    "one start per connection is supported",
                )
            elif kind is not None:
                await self._send("error", f"unsupported message type {kind!r}")

    # ------------------------------------------------------------------ shutdown

    def _request_stop(self, reason: str) -> None:
        """Ask the orchestrator to stop; idempotent and safe from any task."""
        if self._stop_at is None:
            self._stop_at = time.monotonic()
            log.info("web_run_stopping", run_id=self._run_id, reason=reason)
        orchestrator = self._orchestrator
        if orchestrator is not None and self._stop_task is None:
            self._stop_task = asyncio.create_task(
                orchestrator.stop(), name=f"furrow-stop-{self._run_id}"
            )

    def _expire_stop(self) -> None:
        """Cancel the run task when a graceful stop outlasts the grace period."""
        task = self._run_task
        if self._stop_at is None or task is None or task.done():
            return
        if time.monotonic() - self._stop_at <= STOP_GRACE_SECONDS:
            return
        self._stop_at = None
        log.warning(
            "web_run_grace_expired",
            run_id=self._run_id,
            grace_seconds=STOP_GRACE_SECONDS,
        )
        task.cancel()

    def _run_outcome(self) -> tuple[bool | None, str | None]:
        """Completion flag and error text of the run task, once it has finished."""
        task = self._run_task
        if task is None:
            return None, None
        if not task.done() or task.cancelled():
            return None, CANCELLED_REASON
        exc = task.exception()
        if exc is not None:
            return None, f"{type(exc).__name__}: {exc}"
        return bool(task.result()), None

    async def _finish(self) -> None:
        """Stop every task this connection owns, publish the summary, close.

        The bookkeeping (cancelling the tasks, publishing the summary, releasing
        the run slot) happens *before* the first ``await``, so a hard
        cancellation of the endpoint - a server shutdown, for instance - cannot
        skip it and leave a slot or a task behind.

        The single ``done`` frame is the last frame of the connection: the loop
        already reported its own terminal event as :data:`FINISHED_EVENT`, so
        nothing can follow the summary except the close, and a client that acts
        on ``done`` cannot be too late to read it.
        """
        run_task = self._run_task
        watch_task = self._watch_task
        stop_task = self._stop_task
        pending = [task for task in (run_task, watch_task, stop_task) if task is not None]
        for task in pending:
            task.cancel()
        self._publish_summary(*self._run_outcome())
        self.release_slot()
        try:
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            completed, error = self._run_outcome()
            self._publish_summary(completed, error)
            client = self._client
            self._client = None
            if client is not None:
                with contextlib.suppress(Exception):
                    await client.aclose()
            if not self._client_gone and not self._aborted and self._state is not None:
                await self._send(
                    "done",
                    "Goal complete."
                    if completed
                    else (error or "Run finished without completing."),
                    data={
                        "done": bool(completed),
                        "cycles": self._state.cycles,
                        "error": error,
                    },
                )
            await self._close()
        finally:
            self._closed = True

    def _publish_summary(self, completed: bool | None, error: str | None) -> None:
        """Freeze the run state and remember it for ``GET /api/state``."""
        global _last_run
        state = self._state
        if state is None:
            return
        orchestrator = self._orchestrator
        state.active = False
        state.finished_at = _utcnow()
        if orchestrator is not None:
            state.cycles = orchestrator.cycles
            state.done = orchestrator.done
            state.current_plan = _plan_json(orchestrator.current_plan)
            state.last_test_result = _test_result_json(orchestrator.last_test_result)
            state.history = [_plan_json(plan) for plan in orchestrator.history]
            if error is None:
                error = orchestrator.error
        if completed is not None:
            state.done = completed
        state.error = error
        if completed:
            state.outcome = "done"
        elif error == CANCELLED_REASON:
            state.outcome = "cancelled"
        elif error is not None:
            state.outcome = "error"
        elif self._client_gone:
            state.outcome = "client_disconnected"
        else:
            state.outcome = "halted"
        with _state_lock:
            _last_run = state

    def release_slot(self) -> None:
        """Give the concurrency slot back; idempotent."""
        if self._slot_taken:
            self._slot_taken = False
            _release_run_slot()

    async def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            await self._ws.close(code=CLOSE_CODE_NORMAL)


# ------------------------------------------------------------------ http surface


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Serve the single page that drives a run; needs the token when configured."""
    token = _configured_token()
    presented = request.query_params.get("token") or request.headers.get(TOKEN_HEADER)
    if token and not _token_ok(presented):
        return HTMLResponse(_UNAUTHORIZED_HTML, status_code=401)
    return HTMLResponse(_render_index(token), headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness probe. Always open and free of run data, even with auth on."""
    return {
        "status": "ok",
        "version": app_version(),
        "active_runs": _active_run_count(),
        "max_concurrent_runs": _max_concurrent_runs(),
        "web_max_cycles": _web_cycle_budget(),
        "auth_required": web_token_configured(),
        "time": _utcnow(),
    }


@app.get("/api/state")
async def api_state() -> dict[str, Any]:
    """Read-only summary of the most recent run started through the web layer."""
    with _state_lock:
        last = _last_run
    return {
        "version": app_version(),
        "active_runs": _active_run_count(),
        "max_concurrent_runs": _max_concurrent_runs(),
        "web_max_cycles": _web_cycle_budget(),
        "auth_required": web_token_configured(),
        "last_run": last.as_json() if last is not None else None,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Accept one run per connection and stream its events as JSON frames."""
    await websocket.accept()
    if not _acquire_run_slot():
        log.warning("web_run_rejected_busy", active=_active_run_count())
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {
                    "event": "error",
                    "message": (
                        f"the server is already running {_active_run_count()} run(s), "
                        f"the limit is {_max_concurrent_runs()}; try again later"
                    ),
                }
            )
        with contextlib.suppress(Exception):
            await websocket.close(code=CLOSE_CODE_NORMAL)
        return
    session = _WebRun(websocket, slot_taken=True)
    try:
        with contextlib.suppress(WebSocketDisconnect):
            await session.serve()
    except Exception as exc:
        log.warning("websocket_endpoint_failed", error=str(exc), exc_info=True)
        with contextlib.suppress(Exception):
            await websocket.close(code=CLOSE_CODE_NORMAL)
    finally:
        session.release_slot()


# ---------------------------------------------------------------------- entrypoint


def run(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    settings: Settings | None = None,
    log_level: str | None = None,
    max_concurrent_runs: int | None = None,
) -> None:
    """Serve the web UI with uvicorn.

    Synchronous on purpose: it is the process entrypoint and ``uvicorn.run``
    owns the event loop.  *host* defaults to loopback, so exposing the agent
    requires an explicit ``0.0.0.0``.  *settings* carries the web defaults of a
    run (notably ``max_cycles``, see :func:`_web_cycle_budget`).
    """
    global _web_settings, _web_max_concurrent_runs
    if not 1 <= int(port) <= 65535:
        raise ValueError(f"port must be between 1 and 65535, got {port!r}")
    if max_concurrent_runs is not None and int(max_concurrent_runs) < 1:
        raise ValueError(f"max_concurrent_runs must be at least 1, got {max_concurrent_runs!r}")
    if settings is not None:
        _web_settings = settings
    if max_concurrent_runs is not None:
        _web_max_concurrent_runs = int(max_concurrent_runs)
    if log_level is not None:
        configure_logging(log_level)
    log.warning(
        "web_server_starting",
        host=host,
        port=port,
        web_max_cycles=_web_cycle_budget(),
        max_concurrent_runs=_max_concurrent_runs(),
        auth_required=web_token_configured(),
    )
    uvicorn.run(app, host=host, port=port)


# ------------------------------------------------------------------------ the page


def _js_literal(value: str) -> str:
    """Encode *value* as a JS string literal that cannot break out of ``<script>``."""
    encoded = json.dumps(value)
    for raw, escaped in (("<", "\\u003c"), ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        encoded = encoded.replace(raw, escaped)
    return encoded


def _render_index(token: str) -> str:
    return _INDEX_HTML.safe_substitute(
        token_json=_js_literal(token),
        max_goal=MAX_GOAL_LENGTH,
        max_cycles=_web_cycle_budget(),
    )


_INDEX_HTML = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Furrow</title>
<style>
  :root { color-scheme: dark; --fg:#e6e6e6; --bg:#14161a; --panel:#0f1115; --muted:#8a93a0;
          --line:#262b33; --ok:#7ee787; --warn:#e3b341; --err:#f85149; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 1.5rem; background: var(--bg); color: var(--fg);
         font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif; }
  h1 { font-size: 1.2rem; margin: 0 0 .2rem; }
  p.sub { color: var(--muted); margin: 0 0 1rem; }
  form { display: flex; gap: .5rem; flex-wrap: wrap; }
  input { flex: 1 1 22rem; padding: .5rem .65rem; background: var(--panel); color: var(--fg);
          border: 1px solid var(--line); border-radius: 6px; font: inherit; }
  button { padding: .5rem 1rem; border-radius: 6px; border: 1px solid var(--line);
           background: #22262d; color: var(--fg); font: inherit; cursor: pointer; }
  button:disabled { opacity: .45; cursor: not-allowed; }
  #status { margin: .8rem 0; color: var(--muted); }
  #status[data-state="running"] { color: var(--ok); }
  #status[data-state="done"] { color: var(--ok); }
  #status[data-state="warn"] { color: var(--warn); }
  #status[data-state="error"], #status[data-state="halted"],
  #status[data-state="disconnected"] { color: var(--err); }
  #log { list-style: none; margin: 0; padding: .5rem; border: 1px solid var(--line);
         border-radius: 6px; background: var(--panel); max-height: 60vh; overflow: auto;
         font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  #log li { padding: .1rem 0; white-space: pre-wrap; overflow-wrap: anywhere; }
  #log .ev { color: var(--muted); display: inline-block; min-width: 6.5rem; }
  #log li[data-ev="done"] .msg, #log li[data-ev="test"] .msg { color: var(--ok); }
  #log li[data-ev="warn"] .msg { color: var(--warn); }
  #log li[data-ev="error"] .msg { color: var(--err); }
  p.hint { color: var(--muted); font-size: .85rem; }
</style>
</head>
<body>
<h1>Furrow</h1>
<p class="sub">Autonomous coding agent. One run per connection, at most
<span id="budget">$max_cycles</span> cycle(s) per run, goal up to
<span id="limit">$max_goal</span> characters.</p>
<form id="form">
  <input id="goal" maxlength="$max_goal" placeholder="Describe the goal for Furrow"
         autocomplete="off" required>
  <button type="submit" id="start">Start</button>
  <button type="button" id="stop" disabled>Stop</button>
</form>
<p id="status" data-state="idle">Idle.</p>
<ul id="log"></ul>
<p class="hint">Events stream over <code>/ws</code>; the last run is summarised at
<code>/api/state</code>. Press Stop to ask the agent to stop after the current step.</p>
<script>
(function () {
  "use strict";
  var TOKEN = $token_json;
  var MAX_LINES = 500;
  var form = document.getElementById("form");
  var goalInput = document.getElementById("goal");
  var startButton = document.getElementById("start");
  var stopButton = document.getElementById("stop");
  var statusLine = document.getElementById("status");
  var log = document.getElementById("log");
  var socket = null;
  var running = false;

  function setStatus(state, text) {
    statusLine.dataset.state = state;
    statusLine.textContent = text;
  }

  function appendLine(event, message) {
    var item = document.createElement("li");
    item.dataset.ev = event;
    var tag = document.createElement("span");
    tag.className = "ev";
    tag.textContent = "[" + event + "]";
    var body = document.createElement("span");
    body.className = "msg";
    body.textContent = message;
    item.appendChild(tag);
    item.appendChild(body);
    log.appendChild(item);
    while (log.childElementCount > MAX_LINES) {
      log.removeChild(log.firstElementChild);
    }
    log.scrollTop = log.scrollHeight;
  }

  function socketUrl() {
    var scheme = location.protocol === "https:" ? "wss://" : "ws://";
    return scheme + location.host + "/ws";
  }

  function release() {
    var previous = socket;
    socket = null;
    if (!previous) {
      return;
    }
    previous.onopen = null;
    previous.onmessage = null;
    previous.onerror = null;
    previous.onclose = null;
    if (previous.readyState === WebSocket.CONNECTING || previous.readyState === WebSocket.OPEN) {
      previous.close();
    }
  }

  function finish(state, text) {
    running = false;
    startButton.disabled = false;
    stopButton.disabled = true;
    setStatus(state, text);
    release();
  }

  // The final "done" frame is the only one carrying data; report what it says
  // about the run rather than the free-text message, which is the same for every
  // incomplete run. A "done" without data still finishes the page, so a server
  // that omits the payload degrades to the message instead of hanging.
  function outcome(data, text) {
    if (!data || typeof data !== "object") {
      return { state: "done", text: "Finished: " + text };
    }
    var cycles = typeof data.cycles === "number" ? data.cycles : null;
    var spent = cycles === null ? "" : " after " + cycles + " cycle(s)";
    var failure = typeof data.error === "string" && data.error ? data.error : null;
    if (data.done === true) {
      return { state: "done", text: "Goal complete" + spent + "." };
    }
    if (failure) {
      return { state: "error", text: "Failed: " + failure };
    }
    return { state: "halted", text: "Finished without completing" + spent + "." };
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    if (running) {
      return;
    }
    var goal = goalInput.value.trim();
    if (!goal) {
      setStatus("error", "Type a goal first.");
      return;
    }
    log.replaceChildren();
    running = true;
    startButton.disabled = true;
    stopButton.disabled = false;
    setStatus("running", "Connecting...");
    var ws = new WebSocket(socketUrl());
    socket = ws;
    ws.onopen = function () {
      ws.send(JSON.stringify({ type: "start", goal: goal, token: TOKEN }));
    };
    ws.onmessage = function (message) {
      var frame = null;
      try {
        frame = JSON.parse(message.data);
      } catch (err) {
        appendLine("error", "unreadable frame: " + String(err));
        return;
      }
      var name = frame && typeof frame.event === "string" ? frame.event : "event";
      var text = frame && typeof frame.message === "string" ? frame.message : String(message.data);
      appendLine(name, text);
      if (name === "accepted") {
        setStatus("running", "Running (" + (frame.data ? frame.data.run_id : "?") + ")");
      } else if (name === "done") {
        // Only ever the summary frame, so the payload is always still to come.
        var result = outcome(frame.data, text);
        finish(result.state, result.text);
      } else if (name === "error") {
        finish("error", "Error: " + text);
      }
    };
    ws.onerror = function () {
      if (running) {
        setStatus("error", "The WebSocket connection failed.");
      }
    };
    ws.onclose = function (event) {
      if (socket === ws) {
        socket = null;
      }
      if (running) {
        finish("disconnected", "Disconnected before the run finished (code " + event.code + ").");
      }
    };
  });

  stopButton.addEventListener("click", function () {
    if (socket && socket.readyState === WebSocket.OPEN) {
      stopButton.disabled = true;
      socket.send(JSON.stringify({ type: "cancel" }));
      setStatus("running", "Stop requested; finishing the current step...");
    }
  });
})();
</script>
</body>
</html>
""")


_UNAUTHORIZED_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Furrow</title></head>
<body style="font-family: sans-serif">
<h1>401 Unauthorized</h1>
<p>Set <code>FURROW_WEB_TOKEN</code> on the server, then open
<code>/?token=&lt;token&gt;</code> or send the <code>X-Furrow-Token</code> header.</p>
</body>
</html>
"""
