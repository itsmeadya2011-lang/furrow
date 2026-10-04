"""The furrow web tier: a one-page monitoring UI and its WebSocket event stream.

Every line the page shows arrives over ``/ws``.  The previous version of this module
never called ``websocket.send_text``/``send_json``/``send_bytes`` anywhere: the
orchestrator printed progress to a rich ``Console`` on the server's stdout and the
browser sat on ``Starting...`` forever.  The bridge now exists explicitly, and each
of its properties is a way an unbounded agent can waste a machine, so they are worth
stating:

* **A bounded hand-off.**  :class:`_EventBridge` is a fixed-size queue between the
  orchestrator's *synchronous* ``on_event`` sink and the socket sender.  On overflow
  the oldest event is dropped and a single coalesced ``events_dropped`` notice is
  queued - refreshed in place, rather than re-queued, while the client has not taken
  it - so a slow client bounds memory, still receives the newest progress, and never
  blocks the producer, which runs inside the orchestrator's own coroutine, on the
  event loop.
* **One run at a time.**  ``Orchestrator.run`` is unbounded and an agent that edits
  files and shells out is expensive, so :func:`_claim_run` admits one run per process
  and refuses a second socket with a readable message rather than silently doubling
  the resource use.
* **A disconnect stops the run.**  :func:`_relay` watches for the client's
  ``websocket.disconnect`` next to the sender, and the endpoint's ``finally`` cancels
  the orchestrator task and awaits ``aclose()`` on *every* exit path.
* **No traceback on the wire.**  A missing API key, a rate limit or a malformed
  payload produces a structured ``error`` frame plus a close code the client can
  reason about; the traceback is logged server-side.

The endpoint is unauthenticated and lets whoever reaches it drive an agent, which is
why :func:`run` binds loopback by default.  The page itself is XSS-safe by
construction: it writes every value with ``textContent``, never ``innerHTML``,
because the goal and the agent's output are untrusted input.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from io import StringIO
from typing import Any, Final

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError, field_validator
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

app = FastAPI(title="Furrow")

#: Bound on the orchestrator -> socket hand-off queue.  A run emits one event per
#: task per cycle plus a handful per cycle; 256 is far more than any browser absorbs
#: in one loop iteration and small enough that a wedged client cannot grow the
#: process.  Overflow drops the *oldest* event (recent progress is the useful part)
#: and keeps one coalesced ``events_dropped`` notice for the whole backlog.
EVENT_BUFFER_SIZE: Final[int] = 256

#: Normal closure (RFC 6455 7.4.1).
WS_NORMAL_CLOSURE: Final[int] = 1000
#: Policy violation: a malformed request, or a second run while one is in flight.
WS_POLICY_VIOLATION: Final[int] = 1008
#: Internal error: the run itself blew up (no API key, rate limit, network).
WS_INTERNAL_ERROR: Final[int] = 1011

RUN_IN_PROGRESS: Final[str] = (
    "A furrow run is already in progress. Wait for it to finish before starting "
    "another: only one run may drive the agent at a time."
)

#: Run ids currently in flight, per process.  ``_claim_run`` checks and inserts with
#: no ``await`` in between, which is atomic on the single-threaded event loop; an
#: ``asyncio.Lock`` would add nothing there and would bind this registry to the first
#: loop that touched it (every ``TestClient`` portal runs its own).
_active_runs: set[str] = set()
_run_counter: int = 0

#: Queue sentinel meaning "no more events"; never sent to the client.
_SENTINEL: Final = object()


class StartRequest(BaseModel):
    """The JSON object a client sends to ``/ws`` to start a run.

    Kept - and now actually used by :func:`_receive_payload` - so the wire format is
    declared once, with the goal's rules attached, instead of being an unchecked
    ``dict.get`` at the call site.
    """

    goal: str
    model: str | None = None

    @field_validator("goal")
    @classmethod
    def _reject_blank_goal(cls, value: str) -> str:
        """Return the goal stripped, refusing an empty one.

        A blank goal used to reach the orchestrator, where it becomes a
        full-budget agent run planning nothing: the most expensive possible no-op.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("goal must not be empty or whitespace only")
        return stripped


class _EventBridge:
    """Bounded, non-blocking hand-off from ``on_event`` to the socket sender."""

    def __init__(self, maxsize: int = EVENT_BUFFER_SIZE) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
        self._finished = False
        #: How many queue slots have been evicted for being too old to keep.
        self.dropped = 0
        #: The one queued ``events_dropped`` notice, if the client has not taken it yet.
        #: Held so that a later overflow can refresh its count in place instead of
        #: queueing a second notice - that is what "coalesced" means here.
        self._notice: dict[str, Any] | None = None

    def publish(self, event: dict[str, Any]) -> None:
        """Queue *event*.  Called synchronously from inside the orchestrator's loop.

        Never blocks and never raises: the orchestrator already guards its sink, but a
        sink that blocked would stall every agent in the process, and one that raised
        would end a run that is otherwise healthy.

        On overflow the oldest slot goes, and whatever takes its place is chosen by what
        the client is missing: a queued notice is refreshed in place, which costs no
        slot, so the newest event can have it.  Only when there is no notice queued -
        the first overflow, or the one right after the sender took the last notice - is
        the freed slot spent on a fresh notice, and that overflow's event is the one
        lost.  Spending it that way is what keeps a slow client bounded without leaving
        it with nothing but "events dropped" lines.
        """
        if self._finished:
            return
        try:
            self._queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass
        # No ``await`` between the full queue above and this get, so the queue cannot
        # have drained underneath us: dropping the oldest is safe and cannot fail.
        self._queue.get_nowait()
        self.dropped += 1
        if self._notice is not None:
            # The notice is still queued, so refreshing its count where it sits leaves
            # the freed slot free for the newest event.
            self._notice["count"] = self.dropped
            self._queue.put_nowait(event)
            return
        self._notice = {"type": "events_dropped", "count": self.dropped}
        self._queue.put_nowait(self._notice)

    def finish(self) -> None:
        """Queue the terminal sentinel, making room for it if the queue is full."""
        if self._finished:
            return
        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(_SENTINEL)
        self._finished = True

    async def get(self) -> Any:
        """Await the next queued item, which is ``_SENTINEL`` once the run ends.

        A handed-out notice is forgotten here, so the next overflow queues a fresh one
        instead of refreshing a frame the client has already been sent.
        """
        item = await self._queue.get()
        if item is self._notice:
            self._notice = None
        return item


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe: the UI is up, with no dependency on a run being active."""
    return {"status": "ok"}


@app.get("/")
async def index() -> HTMLResponse:
    """The single-page monitoring UI, served inline."""
    return HTMLResponse(content=r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Furrow</title>
<style>
  :root { color-scheme: dark; }
  body {
    margin: 0; padding: 1.5rem; background: #10151c; color: #d7dee7;
    font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  h1 {
    margin: 0 0 1rem; font-size: 1.05rem; letter-spacing: .14em;
    text-transform: uppercase; color: #7fd1a6;
  }
  form { display: flex; gap: .5rem; margin-bottom: 1rem; }
  input {
    flex: 1; padding: .55rem .7rem; font: inherit; color: inherit;
    background: #171d26; border: 1px solid #2b3440; border-radius: 4px;
  }
  button {
    padding: .55rem 1.2rem; font: inherit; cursor: pointer; color: #eafff4;
    background: #21543d; border: 1px solid #2f8f68; border-radius: 4px;
  }
  button[disabled] { opacity: .45; cursor: not-allowed; }
  #out {
    height: 66vh; margin: 0; padding: .8rem; overflow: auto;
    white-space: pre-wrap; word-break: break-word;
    background: #0b0f14; border: 1px solid #222b36; border-radius: 6px;
  }
  .cycle { color: #7fb4ff; font-weight: 600; }
  .task  { color: #e6c07b; }
  .ok    { color: #7fd1a6; }
  .bad   { color: #ff8b8b; }
  .meta  { color: #8b96a3; }
</style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Describe the goal furrow should work on" required />
    <button id="start" type="submit">Start</button>
  </form>
  <pre id="out"></pre>
<script>
const MAX_LINES = 500;   // bounded scrollback: the DOM must not grow without limit
const form = document.getElementById('form');
const goalInput = document.getElementById('goal');
const button = document.getElementById('start');
const out = document.getElementById('out');
const lines = [];
let socket = null;

// textContent, never innerHTML: the goal and the agent's output are untrusted.
function push(cls, text) {
  const line = document.createElement('span');
  line.className = cls;
  line.textContent = text + '\n';
  lines.push(line);
  if (lines.length > MAX_LINES) {
    lines.splice(0, lines.length - MAX_LINES);
  }
  out.textContent = '';
  for (const entry of lines) { out.appendChild(entry); }
  out.scrollTop = out.scrollHeight;
}

function note(text) { push('meta', text); }

function render(event) {
  const type = String(event.type || 'message');
  if (type === 'cycle_start') {
    push('cycle', '── cycle ' + event.cycle + ' ── ' + event.goal);
  } else if (type === 'plan') {
    const tasks = event.tasks || [];
    push('meta', 'plan, ' + tasks.length + ' task(s): ' + event.rationale);
    for (const task of tasks) {
      push('task', '  • ' + task.id + ': ' + task.description);
    }
  } else if (type === 'task_start') {
    push('task', '→ ' + event.task_id + ' started');
  } else if (type === 'task_complete') {
    push('ok', '✓ ' + event.task_id + ': ' + event.summary);
  } else if (type === 'task_failed') {
    push('bad', '✗ ' + event.task_id + ': ' + event.error);
  } else if (type === 'test_result') {
    push(event.passed ? 'ok' : 'bad',
         (event.passed ? 'tests passed: ' : 'tests failed: ') + event.summary);
    for (const failure of event.failures || []) { push('bad', '   - ' + failure); }
  } else if (type === 'goal_updated') {
    push('meta', 'goal updated: ' + event.goal);
  } else if (type === 'events_dropped') {
    push('bad', '!! ' + event.count + ' old event(s) dropped: client too slow');
  } else if (type === 'error') {
    push('bad', 'error: ' + event.message);
  } else if (type === 'done') {
    push('meta', '── done: ' + event.reason + ' after ' + event.cycles + ' cycle(s) ──');
  } else {
    note(type + ': ' + JSON.stringify(event));
  }
}

function setRunning(running) {
  button.disabled = running;
  goalInput.readOnly = running;
}

function stopSocket() {
  const open = socket;
  socket = null;
  setRunning(false);
  if (open) { open.close(); }
}

form.onsubmit = (ev) => {
  ev.preventDefault();
  if (socket) { return; }
  const goal = goalInput.value.trim();
  if (!goal) { note('Enter a goal first.'); return; }
  // Derive the scheme from the page: behind a TLS-terminating proxy this document is
  // https:, and a hardcoded ws: URL is blocked by the browser as mixed content.
  const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
  setRunning(true);
  note('Starting: ' + goal);
  socket = new WebSocket(scheme + '//' + location.host + '/ws');
  // Send only once the handshake completed; sending straight after the constructor
  // is a race that throws on any but the luckiest socket.
  socket.onopen = () => socket.send(JSON.stringify({ goal: goal }));
  socket.onmessage = (msg) => {
    let event;
    try {
      event = JSON.parse(msg.data);
    } catch (err) {
      note('unparsable frame: ' + msg.data);
      return;
    }
    render(event);
    if (event.type === 'done') { stopSocket(); }
  };
  socket.onerror = () => note('socket error: could not reach the furrow server');
  socket.onclose = (ev) => {
    socket = null;
    setRunning(false);
    note('closed (code ' + ev.code + (ev.reason ? ': ' + ev.reason : '') + ')');
  };
};
</script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Open one orchestrator run for one browser and stream it to the socket."""
    await websocket.accept()
    orchestrator: Orchestrator | None = None
    run_task: asyncio.Task[None] | None = None
    run_id: str | None = None
    try:
        try:
            payload = await _receive_payload(websocket)
        except WebSocketDisconnect:
            logger.info("client went away before sending a goal; nothing to run")
            return
        except ValueError as exc:
            await _refuse(websocket, f"Bad request: {exc}")
            return

        run_id = _next_run_id()
        if not _claim_run(run_id):
            await _refuse(websocket, RUN_IN_PROGRESS)
            return

        bridge = _EventBridge()
        # A quiet console: the page is the output surface for a web run, and the
        # server's stdout is not.
        orchestrator = Orchestrator(
            goal=payload.goal,
            console=_quiet_console(),
            on_event=bridge.publish,
        )
        logger.info("run %s started for goal %r", run_id, payload.goal)
        run_task = asyncio.create_task(orchestrator.run(), name=f"furrow-run-{run_id}")
        # The run task is what ends the stream: whatever happens to it - completion,
        # failure, cancellation - it must release the sender.
        run_task.add_done_callback(lambda _task: bridge.finish())

        await _relay(websocket, bridge)
        await _close_after_run(websocket, run_task)
    except Exception:
        logger.exception("unhandled error while streaming run %s", run_id)
        await _send_json(
            websocket,
            {
                "type": "error",
                "message": "The furrow server hit an internal error. See its logs.",
            },
        )
        await _close(websocket, WS_INTERNAL_ERROR)
    finally:
        if run_task is not None:
            await _stop_task(run_task)
        if orchestrator is not None:
            # Idempotent, and safe on a test double with no client to close.
            with contextlib.suppress(Exception):
                await orchestrator.aclose()
        _release_run(run_id)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the UI and its event stream.

    Loopback by default, and deliberately so: ``/ws`` has no authentication, no
    origin check and no rate limiting, and it drives an agent that edits files and
    runs shell commands.  Binding another address publishes that to everything that
    can route to it, so do that only behind an authenticating proxy - the CLI's
    ``furrow web --host`` warns when it is asked to.

    Args:
        host: Address to bind.
        port: TCP port to listen on.
    """
    uvicorn.run(app, host=host, port=port)


# --------------------------------------------------------------------------- input


async def _receive_payload(websocket: WebSocket) -> StartRequest:
    """Parse and validate the goal frame.

    Args:
        websocket: An accepted connection awaiting its first message.

    Returns:
        The validated request.

    Raises:
        WebSocketDisconnect: The client closed before sending a goal.
        ValueError: The frame was not JSON, not an object, or not a usable goal.
            The message is client-safe: pydantic field names and messages, never a
            traceback.
    """
    # Raises on a closed socket, and on a frame that is not decodable JSON.
    raw = await websocket.receive_json()
    if not isinstance(raw, dict):
        raise ValueError(
            f"expected a JSON object with a 'goal' field, got {type(raw).__name__}"
        )
    try:
        return StartRequest.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(_validation_summary(exc)) from exc


def _validation_summary(exc: ValidationError) -> str:
    """Flatten a pydantic error into ``field: message`` pairs on one line."""
    parts = [
        f"{'.'.join(str(part) for part in error['loc']) or '(payload)'}: {error['msg']}"
        for error in exc.errors()
    ]
    return "; ".join(parts) or str(exc)


# ---------------------------------------------------------------------- run registry


def _next_run_id() -> str:
    global _run_counter
    _run_counter += 1
    return f"run-{_run_counter}"


def _claim_run(run_id: str) -> bool:
    """Admit *run_id*, unless a run is already in flight in this process.

    The check and the insert sit next to each other with no ``await`` between them,
    which is what makes this atomic; ``asyncio.Lock`` would be ceremony for the same
    guarantee and would misbehave across event loops.
    """
    if _active_runs:
        logger.info("refusing run %s: %d run(s) already active", run_id, len(_active_runs))
        return False
    _active_runs.add(run_id)
    return True


def _release_run(run_id: str | None) -> None:
    """Free the slot held by *run_id*.  Harmless for a run that never claimed one."""
    if run_id is not None:
        _active_runs.discard(run_id)


# ------------------------------------------------------------------------ streaming


async def _relay(websocket: WebSocket, bridge: _EventBridge) -> None:
    """Stream queued events until the run ends or the client goes away.

    Two coroutines race on purpose.  :func:`_drain` owns the writes; the watcher
    turns "the browser tab closed" into a prompt end of the stream instead of
    waiting for a write that may never fail.  Both are cancelled on the way out, so
    a normal finish never leaves a task parked on ``receive``.
    """
    sender = asyncio.create_task(_drain(websocket, bridge), name="furrow-ws-send")
    watcher = asyncio.create_task(_await_disconnect(websocket), name="furrow-ws-watch")
    try:
        await asyncio.wait({sender, watcher}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        sender.cancel()
        watcher.cancel()
        # ``return_exceptions=True`` retrieves whatever they raised, which is what
        # keeps asyncio quiet about "exception was never retrieved" at teardown.
        await asyncio.gather(sender, watcher, return_exceptions=True)


async def _drain(websocket: WebSocket, bridge: _EventBridge) -> None:
    """Send queued events until the sentinel, or until the socket is unusable."""
    while True:
        item = await bridge.get()
        if item is _SENTINEL:
            return
        if not await _send_json(websocket, item):
            return


async def _await_disconnect(websocket: WebSocket) -> None:
    """Block until the client disconnects.

    Nothing reads from the socket after the goal is parsed, so this has no second
    consumer to race.  It returns rather than raising on purpose: the caller only
    needs "stop now", and the ``finally`` that cancels the run is shared by the
    disconnect and the completion path.
    """
    message = await websocket.receive()
    if message.get("type") == "websocket.disconnect":
        logger.info("client disconnected (code %s); stopping the run", message.get("code"))
        return
    logger.debug("ignoring post-goal frame %r", message.get("type"))


async def _send_json(websocket: WebSocket, message: dict[str, Any]) -> bool:
    """Write one JSON frame.

    Returns:
        ``False`` when the socket is already gone, which ends the stream instead of
        raising: a departed client is an expected outcome, not an error.
    """
    try:
        await websocket.send_json(message)
    except (WebSocketDisconnect, RuntimeError, ConnectionError) as exc:
        logger.debug("dropping a frame for a closed socket: %r", exc)
        return False
    return True


async def _close(websocket: WebSocket, code: int) -> None:
    """Close the socket, tolerating a client that has already gone."""
    try:
        await websocket.close(code=code)
    except (WebSocketDisconnect, RuntimeError, ConnectionError) as exc:
        logger.debug("socket already closed: %r", exc)


async def _refuse(websocket: WebSocket, message: str) -> None:
    """Explain why the request was refused, then close with a policy-violation code."""
    logger.info("refusing websocket request: %s", message)
    await _send_json(websocket, {"type": "error", "message": message})
    await _close(websocket, WS_POLICY_VIOLATION)


async def _close_after_run(websocket: WebSocket, run_task: asyncio.Task[None]) -> None:
    """Report the run's outcome and close: the client's last frames of a run.

    The orchestrator's ``done`` event is the normal terminal message and arrives
    through the queue.  When ``run()`` itself raises, nothing queued a terminal event
    at all, so one is synthesised here with a documented halt reason - the client
    must never be left waiting for a ``done`` that will not come.
    """
    if run_task.cancelled() or not run_task.done():
        # Still running, or cancelled because the client left: nothing to report.
        await _close(websocket, WS_NORMAL_CLOSURE)
        return
    failure = run_task.exception()
    if failure is None:
        await _close(websocket, WS_NORMAL_CLOSURE)
        return
    logger.error(
        "run failed", exc_info=(type(failure), failure, failure.__traceback__)
    )
    await _send_json(
        websocket,
        {"type": "error", "message": f"The run failed: {failure}"},
    )
    await _send_json(
        websocket,
        {"type": "done", "reason": "error", "cycles": 0},
    )
    await _close(websocket, WS_INTERNAL_ERROR)


async def _stop_task(task: asyncio.Task[None]) -> None:
    """Cancel *task* if it is still running, then wait for it to settle.

    Cancelling a finished task is avoided: ``await`` on it would raise
    ``CancelledError`` for work that actually completed.  The wait is also what
    retrieves a task's exception, so asyncio does not report it as never retrieved.
    """
    if not task.done():
        task.cancel()
    # ``return_exceptions=True`` makes a cancelled or failed child a result rather
    # than a raise, while a cancellation of *this* task still propagates.
    await asyncio.gather(task, return_exceptions=True)


def _quiet_console() -> Console:
    """A console that keeps the run's human-facing output away from the server's."""
    return Console(file=StringIO(), no_color=True, highlight=False)