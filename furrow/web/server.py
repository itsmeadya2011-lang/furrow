from __future__ import annotations

import asyncio
import contextlib
import logging

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

logger = logging.getLogger(__name__)

app = FastAPI(title="Furrow")

# Bounded so a browser that stops reading cannot grow the backlog without limit; the
# sender task applies back-pressure by dropping lines the client is too slow to get to.
EVENT_QUEUE_SIZE = 500
# How long we let the sender flush what is already queued once the run is over.
SOCKET_DRAIN_TIMEOUT = 5.0
# The four role fields a ``model`` override replaces: set them all, otherwise the per-role
# defaults in Settings win and the model picked in the UI would silently change nothing.
MODEL_FIELDS = ("model", "planner_model", "worker_model", "tester_model")


def _run_settings(model: str | None) -> Settings:
    """The settings for one request: a copy of the singleton, optionally with an override.

    ``model_copy`` gives the request its own field values, so the override cannot outlive it.
    This module used to assign the override onto the module-level ``settings`` object, which
    leaked one client's model into every later request (a second client that sent no model
    silently inherited the first one's) and let two concurrent clients fight over one global.
    The CLI still mutates the singleton on purpose for its own flags; that is its own
    process, so it needs no isolation and is left alone.
    """
    run_settings = settings.model_copy(deep=True)
    if model:
        for field in MODEL_FIELDS:
            setattr(run_settings, field, model)
    return run_settings


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\\nStarting...\\n';
      // Derived from the page's own scheme: a hardcoded ws:// never connects behind an
      // HTTPS proxy (browsers refuse the mixed-content downgrade).
      const scheme = location.protocol === 'https:' ? 'wss://' : 'ws://';
      const ws = new WebSocket(scheme + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
"""
    )


class _EventSink:
    """The bounded hand-off between the synchronous ``on_event`` callback and the sender.

    The callback cannot await, so it can only ``put_nowait`` against a bounded queue: once a
    client stops reading, the queue fills and something has to go. Dropping the *oldest* line
    to make room for the newest keeps the backlog bounded while guaranteeing that the most
    recent line gets in, so the run's terminal event ("Goal complete") is the last thing lost
    rather than the first -- losing it leaves the page looking stuck. Every drop is counted so
    the loss is diagnosable instead of silent.
    """

    def __init__(self, maxsize: int = EVENT_QUEUE_SIZE) -> None:
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def emit(self, line: str) -> None:
        """Hand one raw text line to the sender task; never block the caller's event loop."""
        if self.queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self.queue.get_nowait()
            self.dropped += 1
        try:
            self.queue.put_nowait(line)
        except asyncio.QueueFull:  # pragma: no cover - unreachable: no await above
            self.dropped += 1


async def _send_events(
    websocket: WebSocket,
    sink: _EventSink,
    disconnected: asyncio.Event,
    finished: asyncio.Event,
) -> None:
    """Own every write to the socket; flag `disconnected` as soon as the client is gone.

    Shutdown is signalled by the dedicated `finished` event, *not* by a ``None`` line pushed
    onto the queue. On a full bounded queue ``put_nowait(None)`` raises ``QueueFull`` and the
    sentinel was silently discarded, so the sender waited on ``queue.get()`` forever, the
    caller burned the whole drain timeout, and the timeout (not the sentinel) was what
    finally stopped the task. ``Event.set()`` has no queue to be too full for, so a full
    queue can no longer strand the sender. Whatever is already queued when `finished` is set
    is flushed before the task returns.
    """
    stop = asyncio.create_task(finished.wait())
    try:
        while True:
            take = asyncio.create_task(sink.queue.get())
            done, _pending = await asyncio.wait({take, stop}, return_when=asyncio.FIRST_COMPLETED)
            if take not in done:
                # Shutting down: send what is left, then stop.
                take.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await take
                while not sink.queue.empty():
                    await websocket.send_text(sink.queue.get_nowait())
                return
            await websocket.send_text(take.result())
    except Exception:  # a dead socket raises whatever the ASGI server raises
        # Swallow it: the orchestrator keeps emitting events and must not see an error
        # raised from under it just because nobody is listening any more.
        disconnected.set()
    finally:
        stop.cancel()


async def _watch_run(
    run_task: asyncio.Task[None],
    sink: _EventSink,
    disconnected: asyncio.Event,
) -> None:
    """Wait for the run to finish or for the client to vanish, whichever happens first."""
    gone = asyncio.create_task(disconnected.wait())
    try:
        await asyncio.wait({run_task, gone}, return_when=asyncio.FIRST_COMPLETED)
        if not run_task.done():
            return  # client left; the caller cancels the orchestrator
        try:
            await run_task
        except Exception as exc:  # report any failure to whoever is listening
            sink.emit(f"Error: run failed: {exc}")
    finally:
        gone.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await gone


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    # Events cross the sync/async boundary through a bounded queue. The orchestrator's
    # on_event callback is synchronous, so it can only enqueue; one sender task owns all
    # socket writes. That keeps send_text off the callback and means a slow or vanished
    # client can never stall the agent loop or raise inside it.
    sink = _EventSink()
    disconnected = asyncio.Event()
    finished = asyncio.Event()
    sender = asyncio.create_task(_send_events(websocket, sink, disconnected, finished))

    def on_event(line: str) -> None:
        # Plain text, markup free, no JSON: the page appends the payload verbatim.
        sink.emit(line)

    goal = ""
    run_task: asyncio.Task[None] | None = None
    try:
        try:
            payload = await websocket.receive_json()
        except (WebSocketDisconnect, ValueError):
            payload = None

        if not isinstance(payload, dict):
            sink.emit("Error: expected a JSON object with a 'goal' field.")
            return

        try:
            request = StartRequest(**payload)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(part) for part in err['loc']) or 'payload'}: {err['msg']}"
                for err in exc.errors()
            )
            sink.emit(f"Error: invalid start request ({details})")
            return

        goal = request.goal.strip()
        if not goal:
            sink.emit("Error: goal must not be empty.")
            return

        # The override lives on this run's own settings copy, so it cannot reach another
        # request; the client is built here so the orchestrator uses those settings too.
        run_task = asyncio.create_task(
            Orchestrator(
                goal=goal,
                client=LLMClient(settings=_run_settings(request.model)),
                on_event=on_event,
            ).run()
        )
        await _watch_run(run_task, sink, disconnected)
    finally:
        try:
            # Stop the agent before touching the socket: a client that left must not keep
            # the orchestrator cycling (and spending tokens) with nobody watching. Only
            # now is the backlog final, so the sender can be told the run is over.
            if run_task is not None:
                run_task.cancel()
                await asyncio.gather(run_task, return_exceptions=True)
        finally:
            # Inner finally so a cancellation part-way through still reaps the sender: no
            # task may outlive this handler on any path.
            if sink.dropped:
                logger.warning(
                    "dropped %d event(s) for goal %r: the client reads slower than the run "
                    "(the queue holds at most %d lines)",
                    sink.dropped,
                    goal,
                    EVENT_QUEUE_SIZE,
                )
            finished.set()  # cannot be dropped, unlike a sentinel pushed onto a full queue
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(sender, timeout=SOCKET_DRAIN_TIMEOUT)
            # wait_for only cancels on timeout: cancel unconditionally so a stalled drain
            # cannot leave the task behind, then reap it.
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            with contextlib.suppress(Exception):  # the peer may already have closed it
                await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
