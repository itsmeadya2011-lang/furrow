from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

# End-of-stream marker put on the event queue once a run has finished.
_SENTINEL = object()

_INDEX_HTML = r"""<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body {
      font-family: system-ui, sans-serif;
      max-width: 64rem;
      margin: 2rem auto;
      padding: 0 1rem;
    }
    form { display: flex; gap: 0.5rem; }
    #goal { flex: 1; }
    #status { font-weight: 600; margin: 0.75rem 0 0.25rem; }
    #log {
      background: #f6f6f6;
      border: 1px solid #ddd;
      padding: 0.75rem;
      min-height: 12rem;
      max-height: 32rem;
      overflow-y: auto;
      white-space: pre-wrap;
    }
    .err { color: #b00020; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <input id="model" placeholder="Model (optional)" />
    <button id="submit" type="submit">Start</button>
  </form>
  <div id="status">Idle</div>
  <pre id="log"></pre>
  <script>
    const form = document.getElementById('form');
    const goalInput = document.getElementById('goal');
    const modelInput = document.getElementById('model');
    const submit = document.getElementById('submit');
    const status = document.getElementById('status');
    const log = document.getElementById('log');

    function line(text, kind) {
      const div = document.createElement('div');
      div.textContent = text;
      if (kind) div.className = kind;
      log.appendChild(div);
      log.scrollTop = log.scrollHeight;
    }

    function setStatus(text) { status.textContent = text; }

    function finish() { submit.disabled = false; }

    function render(frame) {
      switch (frame.type) {
        case 'accepted':
          setStatus('Running (cycle ' + (frame.cycle || 0) + ')');
          line('Goal: ' + frame.goal);
          break;
        case 'cycle_start':
          setStatus('Cycle ' + frame.cycle);
          line('--- Cycle ' + frame.cycle + ' ---');
          break;
        case 'cycle_end':
          line('--- End of cycle ' + frame.cycle + ' ---');
          break;
        case 'plan':
          setStatus('Planned (cycle ' + frame.cycle + ')');
          line('Plan (cycle ' + frame.cycle + '): ' + (frame.rationale || ''));
          (frame.tasks || []).forEach(function (task) {
            let detail = task.description || '';
            if (task.files && task.files.length) {
              detail += ' [files: ' + task.files.join(', ') + ']';
            }
            if (task.dependencies && task.dependencies.length) {
              detail += ' [deps: ' + task.dependencies.join(', ') + ']';
            }
            line('  ' + task.id + '. ' + detail);
          });
          break;
        case 'task_start':
          setStatus('Task ' + frame.task_id);
          line('-> Task ' + frame.task_id + ': ' + (frame.description || ''));
          break;
        case 'task_done':
          line((frame.success ? 'Task ' : 'Task FAILED ') + frame.task_id +
               (frame.summary ? ': ' + frame.summary : ''));
          (frame.files_changed || []).forEach(function (file) {
            line('  changed: ' + file);
          });
          break;
        case 'test_result':
          setStatus(frame.passed ? 'Tests passed' : 'Tests failed');
          line('Tests ' + (frame.passed ? 'passed' : 'FAILED') + ': ' +
               (frame.summary || ''));
          (frame.failures || []).forEach(function (failure) {
            line('  ! ' + failure);
          });
          break;
        case 'done':
          setStatus(frame.reason ? 'Done: ' + frame.reason : 'Done');
          line('Done (' + (frame.reason || 'complete') + ').');
          break;
        case 'error':
          setStatus('Error');
          line('ERROR: ' + frame.message, 'err');
          break;
        case 'log':
          line('[' + (frame.level || 'info') + '] ' + frame.message);
          break;
        default:
          line(JSON.stringify(frame));
      }
    }

    form.onsubmit = async function (event) {
      event.preventDefault();
      const goal = goalInput.value.trim();
      if (!goal) return;
      submit.disabled = true;
      setStatus('Starting...');
      log.textContent = '';
      const payload = { goal: goal };
      const model = modelInput.value.trim();
      if (model) payload.model = model;
      let ws;
      try {
        ws = new WebSocket('ws://' + location.host + '/ws');
        ws.onmessage = function (ev) {
          let frame;
          try {
            frame = JSON.parse(ev.data);
          } catch (err) {
            frame = { type: 'log', level: 'info', message: String(ev.data) };
          }
          render(frame);
          if (frame.type === 'done' || frame.type === 'error') {
            try { ws.close(); } catch (err) { /* already closing */ }
            finish();
          }
        };
        ws.onclose = finish;
        ws.onerror = function () {
          setStatus('Error');
          line('ERROR: websocket connection failed', 'err');
          finish();
        };
        ws.send(JSON.stringify(payload));
      } catch (err) {
        setStatus('Error');
        line('ERROR: ' + err, 'err');
        finish();
      }
    };
  </script>
</body>
</html>
"""


class _QueueLogHandler(logging.Handler):
    """Bridge ``logging`` records onto an ``asyncio.Queue`` as websocket frames.

    Records logged with ``extra={"event": {...}}`` are forwarded verbatim as
    structured frames; every other record becomes a plain
    ``{"type": "log", "level": ..., "message": ...}`` frame. Frames are
    scheduled on the event loop, so the handler is safe to call from worker
    threads. A failure in the bridge must never break the observed run.
    """

    def __init__(self, queue: asyncio.Queue[Any], loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self._queue = queue
        self._loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        try:
            event = getattr(record, "event", None)
            if isinstance(event, dict):
                frame: Any = dict(event)
            else:
                frame = {
                    "type": "log",
                    "level": record.levelname.lower(),
                    "message": record.getMessage(),
                }
            self._loop.call_soon_threadsafe(self._queue.put_nowait, frame)
        except Exception:
            self.handleError(record)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=_INDEX_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    queue: asyncio.Queue[Any] = asyncio.Queue()
    handler = _QueueLogHandler(queue, asyncio.get_running_loop())

    # Route progress through the queue while a run is active. Only the root
    # logger and the "furrow" hierarchy are touched, and the previous levels
    # are restored afterwards.
    root_logger = logging.getLogger()
    furrow_logger = logging.getLogger("furrow")
    previous_root_level = root_logger.level
    previous_furrow_level = furrow_logger.level
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    furrow_logger.setLevel(logging.INFO)

    orchestrator_task: asyncio.Task[None] | None = None
    forwarder_task: asyncio.Task[None] | None = None
    try:
        data = await websocket.receive_json()
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        goal = str(data.get("goal") or "").strip()
        if not goal:
            await websocket.send_json(
                {"type": "error", "message": "goal is required and must be non-empty"}
            )
            return
        model = data.get("model")

        queue.put_nowait({"type": "accepted", "goal": goal, "cycle": 0})
        queue.put_nowait({"type": "cycle_start", "cycle": 1})

        async def run_orchestrator() -> None:
            try:
                orchestrator_kwargs: dict[str, Any] = {}
                if isinstance(model, str) and model.strip():
                    from furrow.config import Settings

                    orchestrator_kwargs["settings"] = Settings(model=model)
                await Orchestrator(goal=goal, **orchestrator_kwargs).run()
                queue.put_nowait({"type": "done", "reason": "complete"})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                queue.put_nowait({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                queue.put_nowait(_SENTINEL)

        async def forward_frames() -> None:
            try:
                while True:
                    frame = await queue.get()
                    if frame is _SENTINEL:
                        return
                    await websocket.send_json(frame)
            except Exception:
                # The socket is gone (client disconnect or send error); stop
                # forwarding so the orchestrator can be cancelled.
                return

        orchestrator_task = asyncio.create_task(run_orchestrator())
        forwarder_task = asyncio.create_task(forward_frames())

        await asyncio.wait({orchestrator_task, forwarder_task}, return_when=asyncio.FIRST_COMPLETED)

        if orchestrator_task.done():
            # The run finished; drain the remaining frames to the socket.
            await forwarder_task
        else:
            # The forwarder stopped early, so the client went away: cancel
            # the run and await its cancellation cleanly.
            orchestrator_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await orchestrator_task
            await forwarder_task
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        if orchestrator_task is not None and not orchestrator_task.done():
            orchestrator_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await orchestrator_task
        if forwarder_task is not None and not forwarder_task.done():
            forwarder_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await forwarder_task
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_root_level)
        furrow_logger.setLevel(previous_furrow_level)
        with contextlib.suppress(Exception):
            await websocket.close()


def run(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    uvicorn.run(app, host=host, port=port, reload=reload)
