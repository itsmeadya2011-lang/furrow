from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.config import settings as global_settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

#: Hard ceiling for a single websocket frame, in UTF-8 bytes. A client that
#: cannot render a frame this size is better served a truncation marker than a
#: buffer blow-up, and it keeps one pathological event from stalling the writer.
MAX_EVENT_BYTES = 8192
#: Per-field / per-collection caps applied before serialisation.
MAX_FIELD_CHARS = 2000
MAX_LIST_ITEMS = 50
MAX_FIELDS = 40
#: How many frames may sit in the bridge queue before the hook starts dropping
#: them. A slow reader must not be able to grow the queue without bound.
EVENT_QUEUE_SIZE = 256
#: The same three role-model fields as ``furrow.cli.main.ROLE_MODEL_FIELDS``.
#: Duplicated rather than imported on purpose: the agents always name their own
#: role model when calling ``LLMClient.complete``, so a model typed into the web
#: form has to be expanded across every role or the override stops at the unused
#: ``model`` field. Importing the CLI module to share the tuple would drag
#: ``click``/``rich`` and the whole command surface into a web-only deployment
#: for three strings, so each front end keeps its own copy of the list.
ROLE_MODEL_FIELDS = ("planner_model", "worker_model", "tester_model")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


def _bound_value(value: Any) -> Any:
    """Coerce an event payload into something small and JSON-serialisable.

    Events are documented as plain dicts of str/int/bool/list, but a hook is a
    public surface: this keeps a surprising value from becoming an
    unserialisable or unbounded frame.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= MAX_FIELD_CHARS:
            return value
        return value[:MAX_FIELD_CHARS] + "…[truncated]"
    if isinstance(value, (list, tuple, set)):
        return [_bound_value(item) for item in list(value)[:MAX_LIST_ITEMS]]
    if isinstance(value, dict):
        items = list(value.items())[:MAX_FIELDS]
        return {str(key): _bound_value(item) for key, item in items}
    text = str(value)
    return text if len(text) <= MAX_FIELD_CHARS else text[:MAX_FIELD_CHARS] + "…[truncated]"


def _frame_size(text: str) -> int:
    """Size of *text* on the wire, in UTF-8 bytes."""
    return len(text.encode("utf-8"))


def _escaped_prefix(frame: str, budget: int) -> str:
    """Longest prefix of *frame* whose JSON form fits *budget* bytes.

    Escaping only ever grows a string (``"`` becomes ``\\"``, ``\\`` becomes
    ``\\\\``, a control character becomes ``\\uXXXX``), so the escaped size grows
    monotonically with the prefix length and a binary search finds the exact cut.
    Slicing by characters also keeps the result valid JSON for multi-byte text.
    """
    low, high, best = 0, len(frame), ""
    while low <= high:
        mid = (low + high) // 2
        candidate = frame[:mid]
        if _frame_size(json.dumps(candidate)) <= budget:
            best, low = candidate, mid + 1
        else:
            high = mid - 1
    return best


def _encode_event(event: dict[str, Any]) -> str:
    """Serialise one event into a frame that is guaranteed to fit the byte cap."""
    payload = _bound_value(dict(event))
    frame = json.dumps(payload, default=str)
    if _frame_size(frame) <= MAX_EVENT_BYTES:
        return frame
    # Still oversized: keep the identifying fields and demote the rest to a
    # detail string, then fit that detail to what is left of the cap. The budget
    # has to be measured on the *escaped* detail: json.dumps expands quotes,
    # backslashes and control characters, so slicing the raw detail against the
    # cap would hand back a frame that overruns it.
    envelope: dict[str, Any] = {
        "type": str(payload.get("type", "event")) if isinstance(payload, dict) else "event",
        "cycle": payload.get("cycle", 0) if isinstance(payload, dict) else 0,
        "truncated": True,
        "detail": "",
    }
    overhead = _frame_size(json.dumps(envelope, default=str))
    if overhead > MAX_EVENT_BYTES:
        # Even an empty detail overflows, so the identifying fields themselves are
        # pathological. Ship a fixed envelope instead of trimming forever.
        return json.dumps({"type": "event", "truncated": True})
    envelope["detail"] = _escaped_prefix(frame, MAX_EVENT_BYTES - overhead)
    return json.dumps(envelope, default=str)


def settings_for_model(model: str | None) -> Settings | None:
    """A per-connection settings copy, or ``None`` when there is nothing to override.

    The process-wide ``furrow.config.settings`` singleton is never mutated: an
    override from one browser tab must not leak into the CLI or another run.

    A supplied model also replaces the three role models, mirroring
    ``furrow.cli.main.build_run_settings`` so that a model typed into the form
    reaches the agents instead of stopping at the unused ``model`` field. An
    empty/omitted model returns ``None`` as before, so "nothing to override" is
    still no copy and the per-role defaults stay in charge.
    """
    if not model:
        return None
    updates: dict[str, Any] = {"model": str(model)}
    for field in ROLE_MODEL_FIELDS:
        updates[field] = str(model)
    return global_settings.model_copy(update=updates)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 2rem; }
    #out { white-space: pre-wrap; word-break: break-word; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <input id="model" placeholder="Model (optional)" />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const button = form.querySelector('button');

    // One frame in, one line out. The server streams a JSON event per message
    // and exactly one terminal frame, so the page never waits on a single blob.
    function render(raw) {
      let ev;
      try { ev = JSON.parse(raw); } catch (e) { return raw; }
      switch (ev.type) {
        case 'cycle_start': return '=== cycle ' + ev.cycle + ': ' + ev.goal;
        case 'plan': return '--- plan (' + ev.count + ' task(s)): ' + ev.rationale;
        case 'task_completed': return '  [' + ev.status + '] ' + ev.id + ': ' + ev.result;
        case 'test_result':
          return '--- tests ' + (ev.passed ? 'passed' : 'FAILED') + ': ' + ev.summary +
                 (ev.failures && ev.failures.length
                   ? '\\n' + ev.failures.map((f) => '    * ' + f).join('\\n')
                   : '');
        case 'cycle_end':
          return '--- cycle ' + ev.cycle + ' end (' + ev.tasks_completed +
                 ' completed, ' + ev.tasks_failed + ' failed)';
        case 'halted': return '--- halted: ' + ev.reason;
        case 'done': return '=== done after ' + ev.cycles + ' cycle(s)';
        case 'error': return '!!! error: ' + ev.message;
        default: return ev.type + ': ' + JSON.stringify(ev);
      }
    }

    form.onsubmit = (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      const model = document.getElementById('model').value;
      out.textContent = 'Starting: ' + goal + '\\n';
      button.disabled = true;

      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => { out.textContent += render(ev.data) + '\\n'; };
      ws.onclose = () => {
        out.textContent += '\\nClosed.\\n';
        button.disabled = false;
      };
      ws.onerror = () => { out.textContent += '\\nConnection error.\\n'; };
      ws.onopen = () => ws.send(JSON.stringify({goal: goal, model: model || null}));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    # ``orchestrator.on_event`` is a *synchronous* callable, so it cannot await a
    # websocket send. The hook only ever does a non-blocking put; a separate task
    # owns every write to the socket.
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=EVENT_QUEUE_SIZE)
    writer: asyncio.Task[None] | None = None

    async def writer_loop() -> None:
        while True:
            frame = await queue.get()
            try:
                if frame is None:
                    return
                try:
                    await websocket.send_text(frame)
                except (WebSocketDisconnect, RuntimeError):
                    # The client is gone; stop writing but leave the run alone.
                    return
            finally:
                queue.task_done()

    def on_event(event: dict[str, Any]) -> None:
        try:
            queue.put_nowait(_encode_event(event))
        except asyncio.QueueFull:
            return  # a slow reader drops frames rather than stalling the loop

    try:
        data = await websocket.receive_json()
        goal = str(data.get("goal", "") or "").strip()
        if not goal:
            await websocket.send_text(json.dumps({"type": "error", "message": "a goal is required"}))
            return

        kwargs: dict[str, Any] = {}
        run_settings = settings_for_model(data.get("model") or None)
        if run_settings is not None:
            kwargs["settings"] = run_settings
        orchestrator = Orchestrator(goal=goal, **kwargs)

        writer = asyncio.create_task(writer_loop())
        orchestrator.on_event = on_event
        try:
            await orchestrator.run()
        finally:
            # Ordering matters: the sentinel is *awaited* so it lands behind every
            # event already queued, and the writer is drained before the terminal
            # frame goes out.
            await queue.put(None)
        await writer
        await websocket.send_text(
            json.dumps(
                {"type": "done", "cycles": int(getattr(orchestrator, "cycles", 0) or 0)}
            )
        )
    except WebSocketDisconnect:
        pass
    # Deliberately blind: the client must always be told how the run ended,
    # whatever the failure was. Narrowing this would leave the browser
    # hanging on a socket that never reports an outcome.
    except Exception as exc:  # noqa: BLE001
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
    finally:
        if writer is not None:
            writer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await writer


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
