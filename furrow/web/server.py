"""HTTP + WebSocket surface for Furrow: REST control plane and live event stream.

Routes
------
``GET  /``                              single-page monitoring UI (no auth)
``GET  /api/health``                    ``{"status": "ok", "version": ...}``
``GET  /api/runs``                      JSON array of active/recent runs
``GET  /api/runs/{run_id}``             ``Orchestrator.snapshot()`` + recent events
``POST /api/runs``                      start a run in the background -> 202 ``{"run_id": ...}``
``POST /api/runs/{run_id}/stop``        request a graceful stop -> 200
``GET  /api/runs/{run_id}/events``      ``{"events": [...], "last_timestamp": float}``
``WS   /ws``                            live event stream (see the two modes below)
``GET  /docs`` ``/redoc`` ``/openapi.json``
                    interactive docs + schema; see "Security" for the token behaviour

``POST /api/runs`` accepts ``{"goal": str, "model": str|null, "max_cycles": int|null}``.
A blank/oversized goal or an out-of-range ``max_cycles`` is rejected with HTTP 400
(validation failures are mapped to 400 on purpose). Unknown run ids give 404, an
unknown or wrong token gives 401, and exceeding ``MAX_ACTIVE_RUNS`` gives 429.

The websocket takes ONE JSON message that selects the mode:

* start mode -- ``{"goal": "...", "model": null, "max_cycles": null}`` creates a run
  and streams every ``Event`` it emits as ``event.to_dict()`` until it halts.
* attach mode -- ``{"run_id": "..."}`` streams an existing run's buffered history
  first, then its live events, and never starts anything.

Control frames are tagged ``"kind": "control"`` and carry no ``data``/``cycle``/
``timestamp``: ``{"type": "started", "run_id": ...}``,
``{"type": "done", "summary": {...}}`` and ``{"type": "error", "error": "..."}``.
Event frames are exactly ``Event.to_dict()``. The tag matters because one event
type is literally named ``error``: branch on ``kind`` (or on the presence of
``data``/``timestamp``), never on ``type`` alone.

Security
--------
This server starts agent runs that execute shell commands inside the workspace and
spend real LLM credits, so treat it as privileged. Two things guard it:

* ``run()`` binds ``Settings.web_host`` (``127.0.0.1`` by default) instead of
  ``0.0.0.0``. Pass ``host=`` explicitly to widen the bind.
* If the environment variable ``FURROW_WEB_TOKEN`` is set to a non-empty value, every
  ``/api/*`` route requires ``Authorization: Bearer <token>`` and the websocket
  handshake requires the same value in a ``?token=`` query parameter (browsers cannot
  set headers on a websocket handshake). ``Settings`` has no field for this token and
  adding one would mean touching ``furrow/config.py``, so it is read from the
  environment directly.
* The interactive docs (``/docs``, ``/redoc``) and the OpenAPI schema
  (``/openapi.json``) are ungated only while no token is configured, which keeps the
  zero-config loopback experience intact. Once ``FURROW_WEB_TOKEN`` is set they are
  closed to everyone: no credential gives 401, exactly like ``/api/*``, and even a
  valid credential gives 404, because they are then not served at all.

  The reason for that last part is a genuine conflict rather than caution. FastAPI
  registers those three routes on the app object itself, so they inherited no auth
  dependency at all and the full API surface stayed publicly readable on a
  token-protected deployment. Gating them behind the bearer header is not enough on
  its own, though: Swagger UI fetches ``openapi.json`` from the same origin in a
  browser-initiated request that carries no ``Authorization`` header, so the page would
  render as a broken "failed to load API definition" shell for the one operator who is
  allowed in. Publishing the schema to a token holder only also undercuts the point of
  the token, since the schema is the map of a surface that starts runs and executes
  shell commands. So the docs are published in exactly the configuration where they are
  safe - no token, loopback only - and refused outright otherwise. Use ``curl`` against
  ``/api/*`` to script against a tokened deployment.

If no token is configured the server logs a prominent warning at startup, because any
host that can reach the port can then start runs and spend credits. Put the server
behind TLS and a reverse proxy (the bundled UI derives ``wss://`` from the page
protocol automatically) rather than exposing it unauthenticated.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from importlib import metadata
from typing import Any

import uvicorn
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from furrow.config import Settings, get_settings, reload_settings
from furrow.core.orchestrator import Orchestrator, RunSummary
from furrow.events import Event, EventBus, EventType

logger = logging.getLogger("furrow.web")

API_PREFIX = "/api"
TOKEN_ENV_VAR = "FURROW_WEB_TOKEN"
FALLBACK_VERSION = "0.1.0"
MAX_GOAL_LENGTH = 4_000
MAX_MODEL_LENGTH = 200
MAX_CYCLES_LIMIT = 10_000
MAX_TRACKED_RUNS = 50
MAX_ACTIVE_RUNS = 8
MAX_EVENTS_IN_RESPONSE = 200
WS_REPLAY_LIMIT = 200
WS_POLL_INTERVAL = 1.0
WS_CLOSE_BAD_PAYLOAD = 1003
WS_CLOSE_POLICY = 1008

WS_INVALID_TOKEN_FRAME = {
    "kind": "control",
    "type": "error",
    "error": "invalid or missing token",
}


def _app_version() -> str:
    try:
        return metadata.version("furrow")
    except Exception:
        return FALLBACK_VERSION


# ---------------------------------------------------------------------------
# request models
# ---------------------------------------------------------------------------


class StartRequest(BaseModel):
    """Body of ``POST /api/runs`` and of a start-mode websocket handshake."""

    goal: str = Field(min_length=1, max_length=MAX_GOAL_LENGTH)
    model: str | None = Field(default=None, max_length=MAX_MODEL_LENGTH)
    max_cycles: int | None = Field(default=None, ge=1, le=MAX_CYCLES_LIMIT)

    @field_validator("goal")
    @classmethod
    def _reject_blank_goal(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("goal must not be blank")
        return stripped

    @field_validator("model")
    @classmethod
    def _strip_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


# ---------------------------------------------------------------------------
# run registry
# ---------------------------------------------------------------------------


@dataclass
class ActiveRun:
    """One tracked run: the orchestrator, its event bus and its background task.

    Subscriber queues deliberately live on the bus (``bus.subscribe()``), not here,
    so that a socket which goes away cannot leave a queue registered.
    """

    id: str
    goal: str
    orchestrator: Orchestrator
    bus: EventBus
    started_at: float = field(default_factory=time.time)
    task: asyncio.Task[RunSummary] | None = None
    summary: RunSummary | None = None
    error: str | None = None

    @property
    def running(self) -> bool:
        """True while the background orchestrator task has not finished."""
        return self.task is not None and not self.task.done()

    def overview(self) -> dict[str, Any]:
        """Return the small record used by ``GET /api/runs``."""
        orchestrator = self.orchestrator
        return {
            "id": self.id,
            "goal": self.goal,
            "running": self.running,
            "cycles": getattr(orchestrator, "cycles", 0),
            "halted_reason": orchestrator.halted_reason,
            "started_at": self.started_at,
        }

    def summary_payload(self) -> dict[str, Any] | None:
        """Return the finished run's summary, or ``None`` while it is still going."""
        return asdict(self.summary) if self.summary is not None else None


class RunRegistry:
    """Bounded, insertion-ordered map of run id -> :class:`ActiveRun`.

    No lock is needed: FastAPI runs every handler and websocket frame on a single
    event loop and none of the methods below awaits, so each mutation is atomic
    with respect to the loop.
    """

    def __init__(self, limit: int = MAX_TRACKED_RUNS) -> None:
        self._limit = max(1, limit)
        self._runs: dict[str, ActiveRun] = {}

    def register(self, run: ActiveRun) -> None:
        self._runs[run.id] = run
        self._evict()

    def get(self, run_id: str) -> ActiveRun | None:
        return self._runs.get(run_id)

    def runs(self) -> list[ActiveRun]:
        """Return every tracked run, oldest first."""
        return list(self._runs.values())

    def clear(self) -> None:
        """Forget every run. Used by tests and by an explicit server restart."""
        self._runs.clear()

    def _evict(self) -> None:
        overflow = len(self._runs) - self._limit
        if overflow <= 0:
            return
        finished = sorted(
            (run for run in self._runs.values() if not run.running),
            key=lambda run: run.started_at,
        )
        for run in finished[:overflow]:
            self._runs.pop(run.id, None)


registry = RunRegistry()


def get_run(run_id: str) -> ActiveRun | None:
    """Return the tracked run with this id, if it is still around."""
    return registry.get(run_id)


def list_runs() -> list[ActiveRun]:
    """Return every tracked run, oldest first."""
    return registry.runs()


def running_run_count() -> int:
    """Return how many tracked runs are still executing."""
    return sum(1 for run in registry.runs() if run.running)


def _settings_for(model: str | None, max_cycles: int | None) -> Settings:
    """Return settings for a new run without mutating the process-wide singleton."""
    overrides: dict[str, Any] = {}
    if model is not None:
        overrides["model"] = model
    if max_cycles is not None:
        overrides["max_cycles"] = max_cycles
    if not overrides:
        return get_settings()
    return reload_settings(**overrides)


def _record_finished(run: ActiveRun, task: asyncio.Task[RunSummary]) -> None:
    """Consume the orchestrator task's outcome so it is never orphaned."""
    if task.cancelled():
        run.summary = None
        return
    error = task.exception()
    if error is not None:
        message = f"{type(error).__name__}: {error}"
        run.error = message
        run.bus.emit(EventType.ERROR, f"Run crashed: {message}")
        logger.error("run %s crashed: %s", run.id, message, exc_info=error)
        return
    run.summary = task.result()


async def create_run(
    goal: str,
    *,
    model: str | None = None,
    max_cycles: int | None = None,
    settings: Settings | None = None,
) -> ActiveRun:
    """Create, register and launch a run in the background.

    The returned record has ``task`` already scheduled; the caller can attach
    websocket subscribers or answer the HTTP request straight away.
    """
    active_settings = settings or _settings_for(model, max_cycles)
    bus = EventBus(active_settings.max_event_history)
    orchestrator = Orchestrator(goal=goal, settings=active_settings, bus=bus)
    # A stand-in orchestrator (tests) may build its own bus; stream whichever it uses.
    effective_bus = getattr(orchestrator, "bus", None)
    if not isinstance(effective_bus, EventBus):
        effective_bus = bus
    run = ActiveRun(
        id=uuid.uuid4().hex[:12],
        goal=goal,
        orchestrator=orchestrator,
        bus=effective_bus,
    )
    registry.register(run)
    run.task = asyncio.create_task(orchestrator.run(), name=f"furrow-run-{run.id}")
    run.task.add_done_callback(lambda task: _record_finished(run, task))
    logger.info("started run %s (goal=%r)", run.id, goal[:80])
    return run


# ---------------------------------------------------------------------------
# authentication
# ---------------------------------------------------------------------------


def configured_token() -> str | None:
    """Return the ``FURROW_WEB_TOKEN`` value, or ``None`` when auth is disabled."""
    raw = os.getenv(TOKEN_ENV_VAR)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def token_matches(provided: str | None) -> bool:
    """Return True when ``provided`` is acceptable, comparing in constant time."""
    expected = configured_token()
    if expected is None:
        return True
    if provided is None:
        return False
    return hmac.compare_digest(provided.strip(), expected)


def _bearer_header(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return value.strip() or None


async def require_auth(request: Request) -> None:
    """Dependency guarding every ``/api`` route when a token is configured."""
    if not token_matches(_bearer_header(request)):
        raise HTTPException(
            status_code=401,
            detail=(
                f"missing or invalid bearer token; send 'Authorization: Bearer <token>' "
                f"when {TOKEN_ENV_VAR} is set"
            ),
            headers={"WWW-Authenticate": "Bearer"},
        )


def log_security_notice(host: str | None = None) -> None:
    """Warn loudly when the privileged surface is reachable without a token."""
    if configured_token() is not None:
        logger.info("Furrow web: API token gate is active (%s is set).", TOKEN_ENV_VAR)
        return
    if os.getenv(TOKEN_ENV_VAR) is not None:
        # F3-R2: the variable IS set, so an operator reading their own shell reasonably
        # believes the gate is closed. `configured_token()` strips it and treats blank or
        # whitespace-only as "no token", so a trailing space in a systemd unit or a `.env`
        # line silently leaves `/api/*` open AND re-opens `/docs`, `/redoc` and
        # `/openapi.json`. Say so at ERROR level, naming the variable, rather than only
        # logging the generic "no authentication" warning the operator has already learned
        # to ignore from the default loopback run.
        logger.error(
            "%s IS SET BUT BLANK (empty or whitespace only): the API token gate is "
            "DISABLED and /api/* is open to anyone who can reach this port on %s.\n"
            "Every run started here executes shell commands inside the workspace and "
            "spends real LLM credits, and /docs, /redoc and /openapi.json are served "
            "unauthenticated because they are only closed while a token is configured.\n"
            "This is almost certainly not what you meant. Unset %s to accept an "
            "unauthenticated server deliberately, or set it to a real secret.",
            TOKEN_ENV_VAR,
            host or get_settings().web_host,
            TOKEN_ENV_VAR,
        )
        return
    logger.warning(
        "Furrow web is serving with NO authentication on %s.\n"
        "Anyone who can reach this port can start agent runs. A run executes shell\n"
        "commands (including test commands) inside the workspace and spends real LLM\n"
        "credits. Bind 127.0.0.1 (the default) or set %s and send\n"
        "'Authorization: Bearer <token>' (websocket: ?token=<token>).",
        host or get_settings().web_host,
        TOKEN_ENV_VAR,
    )


# ---------------------------------------------------------------------------
# application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    log_security_notice()
    yield


# SECURITY: FastAPI's own docs routes are declared on the app object and would carry no
# auth dependency, so they are switched off here and re-declared below behind
# `require_auth` (and refused outright once a token is configured). Leaving them at the
# defaults would publish the whole API surface on a token-protected deployment.
app = FastAPI(
    title="Furrow",
    version=_app_version(),
    lifespan=_lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
router = APIRouter(prefix=API_PREFIX, dependencies=[Depends(require_auth)])


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Report bad request bodies as 400 rather than FastAPI's default 422."""
    return JSONResponse(status_code=400, content={"detail": describe_validation_error(exc)})


def describe_validation_error(exc: ValidationError | RequestValidationError) -> str:
    """Flatten a pydantic/FastAPI validation error into one safe string."""
    parts: list[str] = []
    for error in exc.errors():
        where = ".".join(str(item) for item in error.get("loc", ()) if item != "body")
        message = str(error.get("msg", "invalid value"))
        parts.append(f"{where}: {message}" if where else message)
    return "; ".join(parts) or "invalid request"


def _require_run(run_id: str) -> ActiveRun:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"unknown run id: {run_id}")
    return run


def _event_payloads(bus: EventBus, limit: int = MAX_EVENTS_IN_RESPONSE) -> list[dict[str, Any]]:
    return [event.to_dict() for event in bus.history()[-limit:]]


def _overview_payloads() -> list[dict[str, Any]]:
    return [run.overview() for run in reversed(list_runs())]


@router.get("/health")
async def health() -> dict[str, Any]:
    """Liveness probe plus a little operational context."""
    return {
        "status": "ok",
        "version": _app_version(),
        "runs_tracked": len(list_runs()),
        "runs_running": running_run_count(),
        "auth_required": configured_token() is not None,
    }


@router.get("/runs")
async def list_run_records() -> list[dict[str, Any]]:
    """Return every active/recent run, newest first."""
    return _overview_payloads()


@router.post("/runs", status_code=202)
async def start_run_record(start: StartRequest) -> dict[str, Any]:
    """Start a run in the background and answer immediately with its id."""
    if running_run_count() >= MAX_ACTIVE_RUNS:
        raise HTTPException(
            status_code=429,
            detail=f"too many runs in flight (limit {MAX_ACTIVE_RUNS}); stop one first",
        )
    run = await create_run(start.goal, model=start.model, max_cycles=start.max_cycles)
    return {"run_id": run.id, "goal": run.goal, "started_at": run.started_at}


@router.get("/runs/{run_id}")
async def read_run(run_id: str) -> dict[str, Any]:
    """Return the orchestrator snapshot for a run plus its recent events."""
    run = _require_run(run_id)
    snapshot: dict[str, Any] = dict(run.orchestrator.snapshot())
    return {
        **snapshot,
        "id": run.id,
        "started_at": run.started_at,
        "summary": run.summary_payload(),
        "error": run.error,
        "events": _event_payloads(run.bus),
    }


@router.post("/runs/{run_id}/stop")
async def stop_run_record(run_id: str) -> dict[str, Any]:
    """Ask a run to halt at its next cycle boundary."""
    run = _require_run(run_id)
    await run.orchestrator.stop()
    logger.info("stop requested for run %s", run_id)
    return {
        "run_id": run_id,
        "stopped": True,
        "running": run.running,
        "halted_reason": run.orchestrator.halted_reason,
    }


@router.get("/runs/{run_id}/events")
async def read_run_events(
    run_id: str,
    since: float | None = None,
    limit: int = MAX_EVENTS_IN_RESPONSE,
) -> dict[str, Any]:
    """Return buffered events for polling clients, oldest first.

    ``since`` filters on ``Event.timestamp`` and ``limit`` caps the newest events
    (use ``limit=0`` to fetch nothing but the watermark).
    """
    run = _require_run(run_id)
    history = run.bus.history()
    watermark = history[-1].timestamp if history else 0.0
    events = history if since is None else [e for e in history if e.timestamp > since]
    bounded = max(0, min(limit, MAX_EVENTS_IN_RESPONSE))
    return {
        "run_id": run_id,
        "events": [event.to_dict() for event in events[-bounded:]] if bounded else [],
        "last_timestamp": watermark,
        "running": run.running,
    }


app.include_router(router)


# ---------------------------------------------------------------------------
# interactive docs (public only while no token is configured)
# ---------------------------------------------------------------------------

DOCS_DISABLED_DETAIL = (
    "the interactive docs and the OpenAPI schema are not served while "
    f"{TOKEN_ENV_VAR} is set; browse them from a loopback run with no token, or script "
    "against /api/* instead"
)


def refuse_docs_when_tokened() -> None:
    """404 the docs routes once a token is configured. See the module docstring.

    Called from inside the handler, so ``require_auth`` has already answered 401 for an
    anonymous caller. A valid credential therefore also gets 404: under a token these
    routes are not served at all, rather than served to whoever holds the key.
    """
    if configured_token() is not None:
        raise HTTPException(status_code=404, detail=DOCS_DISABLED_DETAIL)


@app.get("/openapi.json", include_in_schema=False, dependencies=[Depends(require_auth)])
async def openapi_schema() -> JSONResponse:
    """Serve the OpenAPI schema while it is not considered sensitive."""
    refuse_docs_when_tokened()
    return JSONResponse(app.openapi())


@app.get("/docs", include_in_schema=False, dependencies=[Depends(require_auth)])
async def swagger_ui() -> HTMLResponse:
    """Serve the bundled Swagger UI while it is not considered sensitive."""
    refuse_docs_when_tokened()
    # These helpers already build an HTMLResponse; wrapping it again breaks rendering.
    return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{app.title} - docs")


@app.get("/redoc", include_in_schema=False, dependencies=[Depends(require_auth)])
async def redoc_ui() -> HTMLResponse:
    """Serve the bundled ReDoc UI while it is not considered sensitive."""
    refuse_docs_when_tokened()
    return get_redoc_html(openapi_url="/openapi.json", title=f"{app.title} - docs")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the bundled single-page monitoring UI."""
    return HTMLResponse(content=INDEX_HTML)


# ---------------------------------------------------------------------------
# websocket streaming
# ---------------------------------------------------------------------------


async def _safe_send(websocket: WebSocket, payload: dict[str, Any]) -> bool:
    """Send JSON, swallowing the inevitable failures after a client vanishes."""
    try:
        await websocket.send_json(payload)
    except Exception:
        return False
    return True


async def _safe_close(websocket: WebSocket, code: int, reason: str) -> None:
    try:
        await websocket.close(code=code, reason=reason[:120])
    except Exception:
        logger.debug("websocket already closed", exc_info=True)


async def _send_event(websocket: WebSocket, event: Event) -> None:
    """Send one event frame; a peer that has gone away is just a disconnect."""
    try:
        await websocket.send_json(event.to_dict())
    except WebSocketDisconnect:
        raise
    except Exception as exc:
        raise WebSocketDisconnect(code=1006, reason=str(exc)) from exc


async def _reject(websocket: WebSocket, message: str, code: int) -> None:
    await _safe_send(websocket, {"kind": "control", "type": "error", "error": message})
    await _safe_close(websocket, code, message)


async def _run_from_socket(websocket: WebSocket, payload: Any) -> ActiveRun | None:
    """Resolve a handshake to a run, or answer with an error frame and return None."""
    if not isinstance(payload, dict):
        await _reject(websocket, "first message must be a JSON object", WS_CLOSE_BAD_PAYLOAD)
        return None
    requested = payload.get("run_id")
    if isinstance(requested, str) and requested.strip():
        run = get_run(requested.strip())
        if run is None:
            await _reject(websocket, f"unknown run id: {requested}", WS_CLOSE_POLICY)
            return None
        return run
    fields = {key: payload.get(key) for key in ("goal", "model", "max_cycles") if key in payload}
    try:
        start = StartRequest.model_validate(fields)
    except ValidationError as exc:
        await _reject(websocket, describe_validation_error(exc), WS_CLOSE_BAD_PAYLOAD)
        return None
    if running_run_count() >= MAX_ACTIVE_RUNS:
        await _reject(
            websocket,
            f"too many runs in flight (limit {MAX_ACTIVE_RUNS}); stop one first",
            WS_CLOSE_POLICY,
        )
        return None
    return await create_run(start.goal, model=start.model, max_cycles=start.max_cycles)


def _done_payload(run: ActiveRun) -> dict[str, Any]:
    return {
        "kind": "control",
        "type": "done",
        "run_id": run.id,
        "halted_reason": run.orchestrator.halted_reason,
        "summary": run.summary_payload(),
        "error": run.error,
    }


async def _drain(websocket: WebSocket, queue: asyncio.Queue[Event]) -> None:
    """Flush events still sitting in the queue once a run has finished."""
    while True:
        try:
            event = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        await _send_event(websocket, event)


async def _watch_disconnect(websocket: WebSocket) -> None:
    """Return as soon as the peer goes away, ignoring anything else it sends."""
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return


async def _stream_run(
    websocket: WebSocket, run: ActiveRun, queue: asyncio.Queue[Event]
) -> None:
    """Forward bus events to the socket until the run finishes or the peer leaves.

    Waiting for the next event races three outcomes: an event arrives, the client
    disconnected (watched explicitly, because a closed peer does not always make
    ``send`` fail), or the timeout expires so a finished run is noticed while the
    bus is idle instead of hanging forever on an empty queue.
    """
    watcher = asyncio.create_task(
        _watch_disconnect(websocket), name=f"furrow-ws-watch-{run.id}"
    )
    try:
        while True:
            if run.task is not None and run.task.done():
                await _drain(websocket, queue)
                await websocket.send_json(_done_payload(run))
                return
            getter: asyncio.Task[Event] = asyncio.create_task(queue.get())
            try:
                finished, _ = await asyncio.wait(
                    {getter, watcher},
                    timeout=WS_POLL_INTERVAL,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                if not getter.done():
                    getter.cancel()
            if watcher in finished:
                raise WebSocketDisconnect(code=1000)
            if getter not in finished:
                continue
            await _send_event(websocket, getter.result())
    finally:
        watcher.cancel()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Stream a run's events as JSON frames.

    The first client message picks the mode: ``{"goal": ...}`` starts a new run and
    streams it, ``{"run_id": ...}`` attaches to an existing one and streams its
    history followed by live events. Closing the socket detaches the subscriber but
    leaves the run itself alone, so it stays visible and stoppable over the REST API.
    """
    if not token_matches(websocket.query_params.get("token")):
        await _safe_send(websocket, dict(WS_INVALID_TOKEN_FRAME))
        await _safe_close(websocket, WS_CLOSE_POLICY, "invalid or missing token")
        return
    await websocket.accept()
    run: ActiveRun | None = None
    queue: asyncio.Queue[Event] | None = None
    try:
        run = await _run_from_socket(websocket, await websocket.receive_json())
        if run is None:
            return
        if not await _safe_send(
            websocket,
            {
                "kind": "control",
                "type": "started",
                "run_id": run.id,
                "goal": run.goal,
                "running": run.running,
            },
        ):
            return
        # Subscribe before replaying so nothing emitted mid-replay is lost.
        queue = run.bus.subscribe()
        for event in run.bus.history()[-WS_REPLAY_LIMIT:]:
            await _send_event(websocket, event)
        await _stream_run(websocket, run, queue)
    except WebSocketDisconnect:
        logger.debug("websocket disconnected (run=%s)", run.id if run else "-")
    except ValueError as exc:
        # The handshake was not JSON: that is the client's mistake, not a server fault.
        await _reject(websocket, f"expected JSON: {exc}", WS_CLOSE_BAD_PAYLOAD)
    except Exception as exc:
        logger.exception("websocket stream failed")
        await _safe_send(
            websocket,
            {"kind": "control", "type": "error", "error": f"{type(exc).__name__}: {exc}"},
        )
    finally:
        if run is not None and queue is not None:
            run.bus.unsubscribe(queue)
        if run is not None and run.running:
            logger.info("run %s continues without a websocket client", run.id)


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def run(
    host: str | None = None,
    port: int | None = None,
    *,
    settings: Settings | None = None,
    log_level: str | None = None,
) -> None:
    """Serve the web surface with uvicorn.

    Defaults come from ``Settings`` -- ``web_host`` (``127.0.0.1``) and ``web_port``
    (``8000``) -- so the server is not exposed to the network unless asked.
    """
    active = settings or get_settings()
    bind_host = host or active.web_host
    bind_port = active.web_port if port is None else port
    log_security_notice(bind_host)
    logger.info("Furrow web listening on http://%s:%s", bind_host, bind_port)
    uvicorn.run(app, host=bind_host, port=bind_port, log_level=log_level)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Furrow</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 1.25rem;
  background: #0f1115; color: #d8dee9;
  font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
h1 { margin: 0 0 1rem; font-size: 1.15rem; color: #88c0d0; }
form { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: 1rem; }
input, button {
  font: inherit; padding: .45rem .6rem; border-radius: 6px;
  border: 1px solid #3b4252; background: #1b1f27; color: #d8dee9;
}
input { flex: 1 1 12rem; min-width: 0; }
#goal { flex: 3 1 22rem; }
button { cursor: pointer; background: #2f3542; }
button.stop { background: #4b2b2b; }
button:disabled { opacity: .45; cursor: not-allowed; }
#status { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: .75rem; }
#status span { padding: .25rem .55rem; border-radius: 4px; background: #1b1f27; }
#halt.good { color: #a3be8c; }
#halt.bad { color: #bf616a; }
#tasks { margin: 0 0 .75rem; padding: 0; list-style: none; }
#tasks li {
  padding: .25rem .55rem; margin-bottom: .2rem;
  background: #1b1f27; border-left: 3px solid #3b4252;
}
#tasks li.in_progress { border-left-color: #ebcb8b; }
#tasks li.completed { border-left-color: #a3be8c; }
#tasks li.failed { border-left-color: #bf616a; color: #d08770; }
#log {
  height: 45vh; overflow-y: auto; padding: .6rem .75rem; border-radius: 8px;
  background: #0b0d11; border: 1px solid #3b4252;
  white-space: pre-wrap; overflow-wrap: anywhere;
}
#log div { padding: 1px 0; }
.muted { color: #7b88a1; }
.plan { color: #b48ead; }
.tool { color: #81a1c1; }
.ok { color: #a3be8c; }
.fail { color: #d08770; }
.err { color: #bf616a; font-weight: bold; }
</style>
</head>
<body>
<h1>Furrow &mdash; live agent monitoring</h1>
<form id="form">
  <input id="goal" placeholder="Goal, e.g. 'add a /api/health endpoint'" required />
  <input id="model" placeholder="model (optional)" />
  <input id="cycles" type="number" min="1" placeholder="max cycles" />
  <input id="token" type="password" placeholder="token (if required)" />
  <button type="submit" id="start">Start</button>
  <button type="button" id="stop" class="stop" disabled>Stop</button>
</form>
<div id="status">
  <span id="state">idle</span>
  <span id="cycle">cycle 0</span>
  <span id="runid">no run</span>
  <span id="halt"></span>
</div>
<ul id="tasks"></ul>
<div id="log"><div class="muted">Waiting for a goal.</div></div>
<script>
"use strict";
const byId = (id) => document.getElementById(id);
const goalEl = byId("goal");
const modelEl = byId("model");
const cyclesEl = byId("cycles");
const tokenEl = byId("token");
const startBtn = byId("start");
const stopBtn = byId("stop");
const stateEl = byId("state");
const cycleEl = byId("cycle");
const runIdEl = byId("runid");
const haltEl = byId("halt");
const tasksEl = byId("tasks");
const logEl = byId("log");

let socket = null;
let runId = null;
let live = false;
const seen = new Set();
const taskNodes = new Map();

function log(text, cls) {
  const line = document.createElement("div");
  if (cls) { line.className = cls; }
  line.textContent = text;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

function setState(text, bad) {
  stateEl.textContent = text;
  stateEl.className = bad ? "err" : "";
}

function setHalt(reason) {
  haltEl.textContent = reason ? "halted: " + reason : "";
  haltEl.className = reason === "goal_complete" ? "good" : "bad";
}

function setButtons(enabled) {
  startBtn.disabled = enabled;
  goalEl.disabled = enabled;
  stopBtn.disabled = !enabled || !runId;
}

function authHeaders() {
  const value = tokenEl.value.trim();
  return value ? { Authorization: "Bearer " + value } : {};
}

function authQuery() {
  const value = tokenEl.value.trim();
  return value ? "?token=" + encodeURIComponent(value) : "";
}

const STATUSES = ["pending", "in_progress", "completed", "failed"];

function upsertTask(id, description, status) {
  if (!id) { return; }
  let node = taskNodes.get(id);
  if (!node) {
    node = document.createElement("li");
    taskNodes.set(id, node);
    tasksEl.appendChild(node);
  }
  node.className = STATUSES.indexOf(status) >= 0 ? status : "";
  node.textContent = id + ": " + (description || "");
}

function compact(data) {
  try {
    return JSON.stringify(data);
  } catch (err) {
    return "[unserialisable payload]";
  }
}

function onEvent(ev) {
  const key = ev.type + "@" + ev.timestamp;
  if (seen.has(key)) { return; }
  seen.add(key);
  if (ev.cycle) { cycleEl.textContent = "cycle " + ev.cycle; }
  const data = ev.data || {};
  if (ev.type === "cycle_start" || ev.type === "cycle_end") {
    log("[" + ev.type + "] " + ev.message, "muted");
  } else if (ev.type === "plan") {
    log("[plan] " + ev.message, "plan");
  } else if (ev.type === "task_start") {
    upsertTask(data.id, data.description, "in_progress");
    log("[task] " + ev.message);
  } else if (ev.type === "task_end") {
    upsertTask(data.id, data.description, data.status);
    log("[task] " + ev.message + (data.error ? " -- " + data.error : ""),
        data.status === "failed" ? "fail" : "ok");
  } else if (ev.type === "tool_call") {
    log("[tool] " + ev.message + " " + compact(data), "tool");
  } else if (ev.type === "test_result") {
    log("[tests] " + ev.message, data.passed ? "ok" : "fail");
  } else if (ev.type === "error") {
    log("[error] " + ev.message, "err");
  } else if (ev.type === "run_end") {
    log("[run] " + ev.message, "muted");
  } else {
    log("[" + ev.type + "] " + ev.message, "muted");
  }
}

function onSummary(frame) {
  const summary = frame.summary;
  const reason = (summary && summary.halted_reason) || frame.halted_reason || "unknown";
  setState("halted", reason !== "goal_complete");
  setHalt(reason);
  if (summary) { log("[summary] " + compact(summary), "muted"); }
  if (frame.error) { log("[error] " + frame.error, "err"); }
}

function onMessage(raw) {
  let frame;
  try {
    frame = JSON.parse(raw);
  } catch (err) {
    log("[error] unparseable frame from server", "err");
    return;
  }
  if (!frame || typeof frame !== "object") { return; }
  // An event frame is Event.to_dict(): it carries data, cycle and timestamp. One
  // event type is literally named "error", so decide on the shape, never on type.
  if (frame.data !== undefined && frame.timestamp !== undefined) {
    onEvent(frame);
    return;
  }
  if (frame.type === "started") {
    runId = frame.run_id;
    runIdEl.textContent = "run " + runId;
    stopBtn.disabled = false;
    log("[run] started " + runId, "muted");
    return;
  }
  if (frame.type === "done") {
    live = false;
    onSummary(frame);
    setButtons(false);
    return;
  }
  if (frame.type === "error") {
    live = false;
    log("[error] " + (frame.error || "unknown error"), "err");
    setState("failed", true);
    setButtons(false);
    return;
  }
  log("[info] " + compact(frame));
}

function dropSocket() {
  if (!socket) { return; }
  const previous = socket;
  socket = null;
  previous.onopen = null;
  previous.onmessage = null;
  previous.onerror = null;
  previous.onclose = null;
  try {
    previous.close();
  } catch (err) {
    /* already closing */
  }
}

function begin(goal) {
  dropSocket();
  logEl.replaceChildren();
  tasksEl.replaceChildren();
  taskNodes.clear();
  seen.clear();
  runId = null;
  runIdEl.textContent = "no run";
  setHalt("");
  setState("running", false);
  setButtons(true);
  log("Starting: " + goal);
  const payload = {
    goal: goal,
    model: modelEl.value.trim() || null,
    max_cycles: cyclesEl.value ? Number(cyclesEl.value) : null,
  };
  const scheme = location.protocol === "https:" ? "wss://" : "ws://";
  const socket = new WebSocket(scheme + location.host + "/ws" + authQuery());
  live = true;
  socket.onopen = () => {
    log("socket open", "muted");
    socket.send(JSON.stringify(payload));
  };
  socket.onmessage = (message) => { onMessage(message.data); };
  socket.onerror = () => {
    log("[error] websocket error: check that the server is up and the token is right",
        "err");
    setState("socket error", true);
  };
  socket.onclose = (message) => {
    if (live) {
      live = false;
      setState("disconnected", true);
      setHalt("connection lost");
      log("[error] the connection dropped while the run was still going. It may "
          + "still be running: check GET /api/runs, or reload to attach again.",
          "err");
    } else {
      log("socket closed (code " + message.code + ")", "muted");
    }
    setButtons(false);
  };
}

byId("form").addEventListener("submit", (event) => {
  event.preventDefault();
  const goal = goalEl.value.trim();
  if (!goal) {
    log("[error] a goal is required", "err");
    return;
  }
  begin(goal);
});

stopBtn.addEventListener("click", async () => {
  if (!runId) {
    log("[error] no run to stop", "err");
    return;
  }
  stopBtn.disabled = true;
  log("[stop] requesting stop for " + runId, "muted");
  try {
    const response = await fetch("/api/runs/" + encodeURIComponent(runId) + "/stop", {
      method: "POST",
      headers: authHeaders(),
    });
    if (response.ok) {
      log("[stop] stop accepted (HTTP " + response.status + ")", "muted");
    } else {
      log("[error] stop rejected: HTTP " + response.status + " " + await response.text(),
          "err");
    }
  } catch (err) {
    log("[error] stop request failed: " + err, "err");
  }
  if (live) { stopBtn.disabled = false; }
});
</script>
</body>
</html>
"""
