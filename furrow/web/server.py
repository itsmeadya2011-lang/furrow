from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import suppress

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator

from furrow.config import Settings
from furrow.core.events import (
    Event,
    EventBus,
    run_completed,
    run_failed,
    run_started,
)
from furrow.core.orchestrator import Orchestrator
from furrow.errors import FurrowError
from furrow.web.dashboard import build_dashboard

logger = logging.getLogger("furrow.web")

# FastAPI enables the auto-generated docs by default: /docs, /redoc,
# /openapi.json and /docs/oauth2-redirect. Those four routes are registered by
# FastAPI's own ``setup()`` from ``docs_url``/``redoc_url``/``openapi_url``,
# never through this module's handlers, so ``_require_http_auth`` cannot see
# them -- they answered 200 with the full endpoint, schema and parameter
# surface to anyone who could reach the port, even with FURROW_WEB_TOKEN set.
#
# Gating them on the token would mean re-declaring all four routes by hand to
# attach the check, which buys a Swagger UI that a local dashboard has no use
# for: the surface is documented in README.md. Turning them off fails closed
# instead, in keeping with how this module treats risk elsewhere (loopback host
# by default, no token by default, shell access off by default): the four
# paths become unknown routes and 404. To browse the schema locally, drop
# ``docs_url``/``redoc_url``/``openapi_url`` back onto the constructor below
# (and bind to 127.0.0.1) rather than exposing them on a shared interface.
app = FastAPI(
    title="Furrow",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

_WEB_TOKEN: str = os.environ.get("FURROW_WEB_TOKEN", "")
if not _WEB_TOKEN:
    logger.warning(
        "FURROW_WEB_TOKEN is not set -- the Furrow web server is unauthenticated. "
        "It binds to 127.0.0.1 by default, which is safe for local use, but do not "
        "expose it to the network (host=0.0.0.0) without also setting "
        "FURROW_WEB_TOKEN."
    )


def _bearer_matches(authorization: str | None) -> bool:
    if not _WEB_TOKEN or not authorization:
        return False
    scheme, _, value = authorization.partition(" ")
    return scheme.lower() == "bearer" and value == _WEB_TOKEN


def _require_http_auth(request: Request) -> None:
    if _WEB_TOKEN and not _bearer_matches(request.headers.get("authorization")):
        raise HTTPException(status_code=401, detail="unauthorized")


def _ws_token_ok(token: str | None) -> bool:
    if not _WEB_TOKEN:
        return True
    return bool(token) and token == _WEB_TOKEN


class _RunRecord:
    __slots__ = ("bus", "goal", "orchestrator", "task")

    def __init__(self, goal, orchestrator, bus, task):
        self.goal = goal
        self.orchestrator = orchestrator
        self.bus = bus
        self.task = task


_active_runs: dict[str, _RunRecord] = {}


def _register(record: _RunRecord) -> None:
    _active_runs[record.goal] = record


def _unregister(goal: str) -> None:
    _active_runs.pop(goal, None)


async def _drive_run(record: _RunRecord) -> None:
    try:
        await record.orchestrator.run()
    except FurrowError as exc:
        logger.warning("run failed: %s", exc)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("run crashed")
    finally:
        _unregister(record.goal)


def _new_run(goal: str, *, settings: Settings | None = None) -> _RunRecord:
    bus = EventBus()
    orchestrator = Orchestrator(goal=goal, bus=bus, settings=settings)
    record = _RunRecord(goal, orchestrator, bus, None)
    record.task = asyncio.create_task(_drive_run(record))
    _register(record)
    return record


class StartRequest(BaseModel):
    goal: str
    model: str | None = None

    @field_validator("goal")
    @classmethod
    def _nonempty_goal(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("goal must be a non-empty string")
        return value.strip()


class StopRequest(BaseModel):
    action: str | None = None


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    _require_http_auth(request)
    return {"status": "ok"}


@app.get("/api/status")
async def status(request: Request) -> dict[str, object]:
    _require_http_auth(request)
    return {
        "active": len(_active_runs),
        "runs": [
            {"goal": rec.goal, "cycles": rec.orchestrator.cycles,
             "stopped": rec.orchestrator.stopped}
            for rec in _active_runs.values()
        ],
    }


@app.post("/api/start")
async def start(request: Request, body: StartRequest) -> dict[str, object]:
    """Start a run in the background and return immediately.

    A second concurrent run is rejected with HTTP 409; the single-run rule is
    enforced by the registry so a runaway orchestrator cannot stack up.
    """
    _require_http_auth(request)
    if _active_runs:
        raise HTTPException(
            status_code=409,
            detail=f"a run is already active: {next(iter(_active_runs))!r}",
        )
    settings = Settings()
    _new_run(body.goal, settings=settings)
    return {"started": True, "goal": body.goal}


# ---------------------------------------------------------------------------
# WebSocket /ws
# ---------------------------------------------------------------------------
async def _send_safe(websocket: WebSocket, payload: dict[str, object]) -> bool:
    """Send `payload` to the client, returning False if the socket closed."""
    try:
        await websocket.send_json(payload)
        return True
    except (RuntimeError, WebSocketDisconnect):
        return False


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Streaming endpoint: every event the orchestrator emits is forwarded to
    the client as a JSON frame.

    Wiring:
      Orchestrator -> EventBus -> subscriber (this handler) ->
      asyncio.Queue -> websocket.send_json.

    The subscriber is async so it can be awaited by the bus; the queue decouples
    event production from socket I/O so a slow client cannot stall the loop.
    """
    token = websocket.query_params.get("token")
    if not _ws_token_ok(token):
        await websocket.close(code=4401, reason="unauthorized")
        return

    await websocket.accept()

    queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()

    async def _enqueue(event: Event) -> None:
        # The bus awaits this handler, so pushing to the queue must not block on
        # a full socket; put_nowait drops nothing -- the queue is unbounded.
        await queue.put(event.to_dict())

    orchestrator: Orchestrator | None = None
    record: _RunRecord | None = None
    task: asyncio.Task | None = None
    unsubscribe = None

    try:
        raw = await websocket.receive_json()
    except WebSocketDisconnect:
        await websocket.close()
        return
    except json.JSONDecodeError as exc:
        await _send_safe(
            websocket,
            run_failed(f"malformed JSON: {exc.msg}").to_dict(),
        )
        await websocket.close()
        return

    goal = (raw.get("goal") if isinstance(raw, dict) else None) or ""
    if not goal.strip():
        await _send_safe(
            websocket,
            run_failed("goal must be a non-empty string").to_dict(),
        )
        await websocket.close()
        return

    if _active_runs:
        await _send_safe(
            websocket,
                run_failed(
                f"a run is already active: {next(iter(_active_runs))!r}"
            ).to_dict(),
        )
        await websocket.close()
        return

    bus = EventBus()
    orchestrator = Orchestrator(goal=goal.strip(), bus=bus)
    record = _RunRecord(goal.strip(), orchestrator, bus, None)
    task = asyncio.create_task(_drive_run(record))
    record.task = task
    _register(record)

    # Subscribe AFTER the orchestrator exists so the orchestrator's own
    # run_started/run_failed events also reach the client.
    unsubscribe = bus.subscribe(_enqueue)

    await _send_safe(websocket, run_started(goal.strip()).to_dict())

    async def _pump() -> None:
        """Drain the queue until the run ends, then send a terminal frame."""
        assert record is not None
        try:
            while True:
                if record.task.done() and queue.empty():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if item is None:
                    break
                if not await _send_safe(websocket, item):
                    return
        finally:
            # The run finished (or the client disconnected): signal closure.
            await _send_safe(
                websocket,
                run_completed(orchestrator.cycles, "stream closed").to_dict(),
            )

    pump_task = asyncio.create_task(_pump())

    try:
        while True:
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
            except json.JSONDecodeError as exc:
                await _send_safe(
                    websocket,
                    run_failed(f"malformed JSON: {exc.msg}").to_dict(),
                )
                continue
            if isinstance(data, dict) and data.get("action") == "stop":
                orchestrator.request_stop("stopped from web")
    except WebSocketDisconnect:
        if orchestrator is not None:
            orchestrator.request_stop("client disconnected")
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
    finally:
        pump_task.cancel()
        with suppress(asyncio.CancelledError):
            await pump_task
        if unsubscribe is not None:
            unsubscribe()
        if record is not None:
            _unregister(record.goal)
        with suppress(Exception):
            await websocket.close()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
_DASHBOARD_HTML = build_dashboard(_WEB_TOKEN)


@app.get("/")
async def index(request: Request) -> HTMLResponse:
    """Serve the dashboard.

    The page is a single self-contained HTML string -- no static-asset
    directory, no build step. All user-supplied text (goal, model output,
    task descriptions) is injected via textContent / createElement so model
    output can never execute as script. See the JS below for the full event
    rendering pipeline.

    Guarded like the other HTTP endpoints when FURROW_WEB_TOKEN is set.
    The token is accepted as ?token= as well as a bearer header, for the
    same reason /ws takes it: a browser cannot set headers on a plain
    page load, and the dashboard's WebSocket handshake needs the token in
    the URL anyway.
    """
    if not _ws_token_ok(request.query_params.get("token")):
        _require_http_auth(request)
    return HTMLResponse(content=_DASHBOARD_HTML)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the Furrow web server.

    Defaults to loopback-only with no token. To expose beyond the machine,
    set FURROW_WEB_TOKEN and pass host="0.0.0.0" explicitly -- see the
    security comment near the top of this module.
    """
    uvicorn.run(app, host=host, port=port)
