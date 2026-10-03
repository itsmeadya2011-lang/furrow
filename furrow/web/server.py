"""HTTP + websocket surface for Furrow.

The orchestrator reports progress on an in-process :class:`~furrow.core.events.EventBus`.
This module owns the plumbing that turns that bus into a websocket stream, and it
never lets agent output reach the page as markup: every line the browser renders
arrives as JSON and is inserted with ``textContent``.

Only one run is in flight at a time.  The run slot, the subscriber queue and the
task wrapping :meth:`Orchestrator.run` are all released in ``finally`` blocks so a
crashed run, a vanished browser or a double submit can never wedge the server.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings, settings
from furrow.core.events import Event, EventBus
from furrow.core.orchestrator import Orchestrator
from furrow.state import RunState, StateStore

logger = logging.getLogger("furrow.web")

#: How many runs ``GET /api/runs`` reports, newest first.
MAX_LISTED_RUNS = 20
#: Upper bound on retained run records so a long lived server cannot grow forever.
MAX_RETAINED_RUNS = 50


def _resolve_version() -> str:
    """Prefer the package attribute, fall back to installed metadata, then a default."""
    try:
        from furrow import __version__ as package_version
    except ImportError:  # pragma: no cover - only on a checkout without the attribute
        try:
            return metadata.version("furrow")
        except metadata.PackageNotFoundError:  # pragma: no cover - not installed
            return "0.2.0"
    return package_version


__version__ = _resolve_version()

app = FastAPI(title="Furrow", version=__version__)


class StartRequest(BaseModel):
    """Body of ``POST /api/runs`` and of the websocket start frame."""

    goal: str
    model: str | None = None
    max_cycles: int | None = None


class RunActiveError(RuntimeError):
    """Raised when a start is requested while another run still holds the slot."""


class _EndOfStream:
    """Sentinel queued to subscribers when a run finishes, so sockets stop waiting."""


_END_OF_STREAM = _EndOfStream()


@dataclass
class RunRecord:
    """Everything the API and the sockets need to know about one run."""

    run_id: str
    orchestrator: Orchestrator
    started_at: str
    # The wrapper coroutine records its outcome on ``state``; it returns nothing.
    task: asyncio.Task[None] | None = None
    done: bool = False
    state: RunState | None = None
    subscribers: list[asyncio.Queue[Any]] = field(default_factory=list)


class RunManager:
    """Owns the run records and the single in-flight run slot."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}

    @property
    def active(self) -> RunRecord | None:
        """The run currently holding the slot, if any."""
        for record in reversed(self._runs.values()):
            if record.done or (record.task is not None and record.task.done()):
                continue
            return record
        return None

    def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def recent(self, limit: int = MAX_LISTED_RUNS) -> list[RunRecord]:
        """Most recently started runs first."""
        return list(reversed(self._runs.values()))[:limit]

    async def start(
        self,
        goal: str,
        model: str | None = None,
        max_cycles: int | None = None,
    ) -> RunRecord:
        """Start a run, or raise :class:`RunActiveError` if the slot is taken.

        There is no ``await`` between the slot check and the assignment below, so
        two concurrent requests cannot both win the slot.
        """
        if self.active is not None:
            raise RunActiveError("a run is already active")

        run_settings = _settings_for(model=model, max_cycles=max_cycles)
        orchestrator = Orchestrator(
            goal=goal,
            settings=run_settings,
            store=StateStore(run_settings.resolved_state_path),
            events=EventBus(),
        )
        record = RunRecord(
            run_id=uuid4().hex[:12],
            orchestrator=orchestrator,
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._runs[record.run_id] = record
        self._prune()
        record.task = asyncio.create_task(self._execute(record), name=f"furrow-run-{record.run_id}")
        return record

    async def stop(self, run_id: str) -> bool:
        """Ask a run to wind down. ``False`` when the id is unknown or already finished."""
        record = self.get(run_id)
        if record is None or record.done:
            return False
        await record.orchestrator.stop()
        return True

    def subscribe(self, record: RunRecord) -> asyncio.Queue[Any]:
        queue = record.orchestrator.events.subscribe()
        record.subscribers.append(queue)
        return queue

    def unsubscribe(self, record: RunRecord, queue: asyncio.Queue[Any]) -> None:
        record.orchestrator.events.unsubscribe(queue)
        if queue in record.subscribers:
            record.subscribers.remove(queue)

    async def _execute(self, record: RunRecord) -> None:
        """Run the orchestrator as a task, never letting its failure disappear."""
        events = record.orchestrator.events
        try:
            record.state = await record.orchestrator.run()
        except asyncio.CancelledError:
            events.emit(Event(kind="error", message="run cancelled"))
            raise
        except Exception as exc:  # noqa: BLE001 - reported to the UI, never swallowed silently
            logger.exception("run %s failed", record.run_id)
            events.emit(Event(kind="error", message=f"{type(exc).__name__}: {exc}"))
        finally:
            record.done = True
            for queue in list(record.subscribers):
                queue.put_nowait(_END_OF_STREAM)

    def _prune(self) -> None:
        if len(self._runs) <= MAX_RETAINED_RUNS:
            return
        for run_id, record in list(self._runs.items()):
            if len(self._runs) <= MAX_RETAINED_RUNS:
                break
            if record.done:
                del self._runs[run_id]


#: Process wide manager: one slot, one set of records.
manager = RunManager()


def _settings_for(model: str | None = None, max_cycles: int | None = None) -> Settings:
    """Copy the global settings with per-run overrides. Never mutates the singleton."""
    overrides: dict[str, Any] = {}
    if isinstance(model, str) and model.strip():
        overrides["model"] = model.strip()
    if max_cycles is not None:
        try:
            overrides["max_cycles"] = int(max_cycles)
        except (TypeError, ValueError):
            logger.warning("ignoring non-numeric max_cycles %r", max_cycles)
    if not overrides:
        return settings
    return settings.model_copy(update=overrides)


def _state_payload(state: RunState | None, *, goal: str = "") -> dict[str, Any]:
    """Run state as JSON, degrading to the bare goal before the first save."""
    if state is None:
        return {"goal": goal}
    return state.model_dump(mode="json")


def _run_summary(record: RunRecord) -> dict[str, Any]:
    state = record.state
    return {
        "run_id": record.run_id,
        "goal": state.goal if state is not None else record.orchestrator.goal,
        "done": record.done,
        "cycles": state.cycles if state is not None else 0,
        "halted_reason": state.halted_reason if state is not None else None,
        "started_at": record.started_at,
    }


async def _send(websocket: WebSocket, payload: dict[str, Any], *, close: bool = True) -> None:
    try:
        await websocket.send_text(json.dumps(payload, default=str))
        if close:
            await websocket.close()
    except (WebSocketDisconnect, RuntimeError):  # pragma: no cover - client already gone
        logger.debug("socket closed before payload could be sent")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/runs", status_code=status.HTTP_202_ACCEPTED)
async def create_run(request: StartRequest) -> dict[str, Any]:
    try:
        record = await manager.start(
            goal=request.goal,
            model=request.model,
            max_cycles=request.max_cycles,
        )
    except RunActiveError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="a run is already active"
        ) from None
    return {"run_id": record.run_id, "state": _state_payload(record.state, goal=request.goal)}


@app.get("/api/runs")
async def list_runs() -> dict[str, Any]:
    return {"runs": [_run_summary(record) for record in manager.recent()]}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    record = manager.get(run_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown run")
    return _state_payload(record.state, goal=record.orchestrator.goal)


@app.post("/api/runs/{run_id}/stop")
async def stop_run(run_id: str) -> dict[str, bool]:
    if manager.get(run_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown run")
    return {"stopped": await manager.stop(run_id)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Stream one run's events, then its final state, then close.

    Attach with ``?run_id=<id>`` to follow a run started over REST, or send
    ``{"goal": ..., "model": ..., "max_cycles": ...}`` as the first frame to start
    one here.
    """
    await websocket.accept()
    record: RunRecord | None = None
    queue: asyncio.Queue[Any] | None = None
    try:
        record, queue = await _attach(websocket)
        if record is None or queue is None:
            return
        await _stream(websocket, record, queue)
    except WebSocketDisconnect:
        logger.debug("client left run %s", record.run_id if record else "?")
    except Exception as exc:  # noqa: BLE001 - never leak a traceback into the socket loop
        logger.exception("websocket session failed")
        detail = f"{type(exc).__name__}: {exc}"
        await _send(websocket, {"kind": "error", "message": detail}, close=False)
    finally:
        if record is not None and queue is not None:
            manager.unsubscribe(record, queue)


async def _attach(websocket: WebSocket) -> tuple[RunRecord | None, asyncio.Queue[Any] | None]:
    """Resolve which run this socket follows and subscribe to it."""
    run_id = websocket.query_params.get("run_id")
    if run_id:
        record = manager.get(run_id)
        if record is None:
            await _send(websocket, {"kind": "error", "message": "unknown run"})
            return None, None
        return record, manager.subscribe(record)

    frame = await websocket.receive_json()
    if not isinstance(frame, dict):
        await _send(websocket, {"kind": "error", "message": "a goal is required"})
        return None, None
    goal = str(frame.get("goal") or "").strip()
    if not goal:
        await _send(websocket, {"kind": "error", "message": "a goal is required"})
        return None, None
    try:
        record = await manager.start(
            goal=goal,
            model=frame.get("model"),
            max_cycles=frame.get("max_cycles"),
        )
    except RunActiveError:
        await _send(websocket, {"kind": "error", "message": "a run is already active"})
        return None, None
    return record, manager.subscribe(record)


async def _stream(websocket: WebSocket, record: RunRecord, queue: asyncio.Queue[Any]) -> None:
    """Forward events until the run ends or the client disappears."""
    gone = asyncio.create_task(_watch_disconnect(websocket), name="furrow-ws-watch")
    try:
        while True:
            if record.done and queue.empty():
                break
            getter = asyncio.create_task(queue.get(), name="furrow-ws-next")
            await asyncio.wait({getter, gone}, return_when=asyncio.FIRST_COMPLETED)
            if gone.done():
                getter.cancel()
                return
            event = getter.result()
            if event is _END_OF_STREAM:
                break
            await websocket.send_text(json.dumps(event.as_dict(), default=str))
        await websocket.send_text(
            json.dumps(
                {
                    "kind": "state",
                    "run_id": record.run_id,
                    "state": _state_payload(record.state, goal=record.orchestrator.goal),
                },
                default=str,
            )
        )
    finally:
        gone.cancel()
    try:
        await websocket.close()
    except RuntimeError:  # pragma: no cover - already closed by the peer
        logger.debug("socket for run %s already closed", record.run_id)


async def _watch_disconnect(websocket: WebSocket) -> None:
    """Resolve as soon as the peer goes away, so a dead tab cannot pin a queue."""
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
    except (WebSocketDisconnect, RuntimeError):  # pragma: no cover - peer vanished
        return


def run(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Serve the web UI.

    Loopback is the default on purpose: a run executes model written code in the
    workspace, so binding every interface has to be an explicit choice.
    """
    if reload:
        # uvicorn can only watch for changes when it owns the import.
        uvicorn.run("furrow.web.server:app", host=host, port=port, reload=True)
    else:
        uvicorn.run(app, host=host, port=port)


# Dependency free page: inline styles, inline script, no CDN.
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Furrow</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 1.5rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: #12141a; color: #e6e8ec;
}
h1 { margin: 0 0 1rem; font-size: 1.05rem; letter-spacing: .25em; color: #7ee787; }
form { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: .6rem; }
input[type="text"] {
  flex: 1 1 22rem; padding: .55rem .7rem; font: inherit; color: inherit;
  background: #1b1e26; border: 1px solid #30363d; border-radius: 4px;
}
button {
  padding: .55rem .95rem; font: inherit; color: inherit; cursor: pointer;
  background: #21262d; border: 1px solid #30363d; border-radius: 4px;
}
button:disabled { opacity: .4; cursor: not-allowed; }
#meta { min-height: 1.2em; margin-bottom: .5rem; font-size: .8rem; color: #9aa4b2; }
#out {
  margin: 0; padding: .8rem; min-height: 20rem; max-height: 68vh; overflow: auto;
  background: #0c0e13; border: 1px solid #30363d; border-radius: 4px;
  font-size: .85rem; line-height: 1.45; white-space: pre-wrap; word-break: break-word;
}
</style>
</head>
<body>
<h1>FURROW</h1>
<form id="form">
  <input id="goal" type="text" placeholder="Describe what Furrow should build" required />
  <button type="submit" id="start">Start</button>
  <button type="button" id="stop" disabled>Stop</button>
</form>
<div id="meta">idle</div>
<pre id="out"></pre>
<script>
(function () {
  "use strict";

  var form = document.getElementById("form");
  var goalInput = document.getElementById("goal");
  var startBtn = document.getElementById("start");
  var stopBtn = document.getElementById("stop");
  var out = document.getElementById("out");
  var meta = document.getElementById("meta");
  var runId = null;
  var socket = null;

  // Server data only ever reaches the page through textContent.
  function line(text) {
    out.textContent += text + "\n";
    out.scrollTop = out.scrollHeight;
  }

  function setRunId(id) {
    runId = id || null;
    meta.textContent = runId ? "run " + runId : "idle";
    stopBtn.disabled = !runId;
  }

  function setBusy(busy) {
    startBtn.disabled = busy;
    goalInput.disabled = busy;
    if (busy) {
      stopBtn.disabled = !runId;
    } else {
      stopBtn.disabled = true;
    }
  }

  function pairs(data, skip) {
    var parts = [];
    if (!data || typeof data !== "object") { return parts; }
    Object.keys(data).forEach(function (key) {
      if (skip.indexOf(key) !== -1) { return; }
      var value = data[key];
      if (value === null || value === undefined || value === "") { return; }
      if (typeof value === "object") { return; }
      parts.push(key + "=" + String(value));
    });
    return parts;
  }

  function describe(ev) {
    var data = ev.data || {};
    var kind = ev.kind || "event";
    var skip = [];
    var text = "[" + kind + "]";
    if (ev.cycle) { text += " cycle " + ev.cycle; }
    if (kind === "task_start" || kind === "task_end") {
      var id = data.task_id || data.id;
      if (id) { text += " task " + id; }
      if (data.status) { text += " -> " + data.status; skip.push("status"); }
      skip.push("task_id", "id");
    }
    if (kind === "test_result") {
      if (typeof data.passed === "boolean") {
        text += data.passed ? " PASS" : " FAIL";
        skip.push("passed");
      }
      if (data.summary) { text += ": " + data.summary; skip.push("summary"); }
    }
    if (ev.message) { text += " " + ev.message; }
    var extra = pairs(data, skip);
    if (extra.length) { text += " {" + extra.join(", ") + "}"; }
    return text;
  }

  function describeState(state) {
    var lines = ["", "* run finished"];
    if (state.goal) { lines.push("  goal: " + state.goal); }
    lines.push("  cycles: " + (state.cycles || 0));
    lines.push("  done: " + Boolean(state.done));
    if (state.halted_reason) { lines.push("  halted: " + state.halted_reason); }
    (state.tasks || []).forEach(function (task) {
      lines.push("  task " + (task.id || "?") + ": " + (task.status || "?"));
    });
    return lines.join("\n");
  }

  function connect(id) {
    var scheme = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(scheme + "//" + location.host + "/ws?run_id=" +
                           encodeURIComponent(id));
    socket.addEventListener("message", function (msg) {
      var ev = null;
      try { ev = JSON.parse(msg.data); } catch (err) { line(String(msg.data)); return; }
      if (ev.kind === "state") {
        setRunId(ev.run_id);
        line(describeState(ev.state || {}));
      } else if (ev.kind === "error") {
        line("[error] " + (ev.message || "unknown error"));
      } else {
        line(describe(ev));
      }
    });
    socket.addEventListener("close", function () {
      line("* closed");
      socket = null;
      setBusy(false);
    });
    socket.addEventListener("error", function () { line("* socket error"); });
  }

  function post(path) {
    return fetch(path, { method: "POST" }).then(function (res) {
      if (!res.ok) { line("! " + path + " -> HTTP " + res.status); }
      return res;
    }, function (err) { line("! " + path + " -> " + err); });
  }

  stopBtn.addEventListener("click", function () {
    if (!runId) { return; }
    stopBtn.disabled = true;
    line("* stopping " + runId);
    post("/api/runs/" + encodeURIComponent(runId) + "/stop");
  });

  form.addEventListener("submit", function (evt) {
    evt.preventDefault();
    if (socket) { return; }
    var goal = goalInput.value.trim();
    if (!goal) { return; }
    setBusy(true);
    setRunId(null);
    line("* starting run");
    fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ goal: goal })
    }).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (body) {
        if (!res.ok) {
          line("! " + (body.detail || ("HTTP " + res.status)));
          setBusy(false);
          return null;
        }
        return body;
      });
    }).then(function (body) {
      if (!body || !body.run_id) { setBusy(false); return; }
      setRunId(body.run_id);
      line("* run " + body.run_id + " accepted");
      connect(body.run_id);
    }, function (err) {
      line("! could not start the run: " + err);
      setBusy(false);
    });
  });
})();
</script>
</body>
</html>
"""
