"""HTTP + WebSocket front end for furrow.

``GET /`` serves a self-contained page that drives a run over the ``/ws``
WebSocket: the browser sends ``{"goal": ..., "model": ...}`` once the socket is
open and receives one JSON-encoded
:class:`~furrow.core.events.Event` per line until the run ends. ``/api/health``
and ``/api/config`` expose non-sensitive runtime information for probes.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow import __version__
from furrow.config import RunResult, Settings, settings
from furrow.core.events import Event, EventKind, make_event
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

__all__ = ["app", "run"]

app = FastAPI(title="Furrow")

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Furrow</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 44rem; padding: 0 1rem; }
  h1 { margin-bottom: 0.25rem; }
  p.hint { color: #8b949e; margin-top: 0; }
  label { display: block; margin: 0.75rem 0 0.25rem; font-size: 0.85rem; color: #8b949e; }
  input { width: 100%; box-sizing: border-box; padding: 0.5rem; font-size: 1rem;
          background: #0d1117; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px; }
  button { margin-top: 1rem; padding: 0.55rem 1.25rem; font-size: 1rem; cursor: pointer;
           border: 0; border-radius: 6px; background: #238636; color: #fff; }
  button:disabled { background: #30363d; color: #8b949e; cursor: progress; }
  #status { margin: 1rem 0 0.5rem; color: #58a6ff; }
  #log { background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
         padding: 0.75rem; height: 26rem; overflow: auto; white-space: pre-wrap;
         word-break: break-word; font-size: 0.85rem; }
</style>
</head>
<body>
  <h1>Furrow</h1>
  <p class="hint">Autonomous coding agent. Enter a goal and watch it work.</p>
  <form id="run">
    <label for="goal">Goal</label>
    <input id="goal" name="goal" type="text" placeholder="Describe what to build" required autocomplete="off" />
    <label for="model">Model (optional)</label>
    <input id="model" name="model" type="text" placeholder="Leave empty for the configured default" autocomplete="off" />
    <button id="start" type="submit">Start</button>
  </form>
  <p id="status">Idle.</p>
  <pre id="log"></pre>
  <script>
    const form = document.getElementById('run');
    const goalInput = document.getElementById('goal');
    const modelInput = document.getElementById('model');
    const button = document.getElementById('start');
    const statusLine = document.getElementById('status');
    const log = document.getElementById('log');

    function append(line) {
      log.textContent += line + '\\n';
      log.scrollTop = log.scrollHeight;
    }

    function render(event) {
      const parts = [];
      if (event.cycle) { parts.push('cycle ' + event.cycle); }
      parts.push(event.kind || 'event');
      if (event.task_id) { parts.push(event.task_id); }
      if (event.message) { parts.push(event.message); }
      return parts.join(' · ');
    }

    form.addEventListener('submit', (submitEvent) => {
      submitEvent.preventDefault();
      const goal = goalInput.value.trim();
      if (!goal) {
        statusLine.textContent = 'Enter a goal first.';
        return;
      }
      const model = modelInput.value.trim();
      let lastKind = '';
      button.disabled = true;
      statusLine.textContent = 'Running...';
      append('> ' + goal + (model ? '  [' + model + ']' : ''));
      const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(scheme + '//' + location.host + '/ws');
      ws.onopen = () => ws.send(JSON.stringify({goal: goal, model: model || null}));
      ws.onmessage = (messageEvent) => {
        let event;
        try {
          event = JSON.parse(messageEvent.data);
        } catch (err) {
          append('[unparsable] ' + messageEvent.data);
          return;
        }
        lastKind = event.kind || '';
        append(render(event));
      };
      ws.onerror = () => { statusLine.textContent = 'Connection error.'; };
      ws.onclose = () => {
        button.disabled = false;
        if (lastKind === 'done') {
          statusLine.textContent = 'Finished.';
        } else if (lastKind === 'error') {
          statusLine.textContent = 'Run failed.';
        } else {
          statusLine.textContent = 'Disconnected.';
        }
      };
    });
  </script>
</body>
</html>
"""


class StartRequest(BaseModel):
    """Payload accepted by the ``/ws`` endpoint."""

    goal: str
    model: str | None = None


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single page UI that drives runs over the WebSocket."""
    return HTMLResponse(content=INDEX_HTML)


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Report that the server is up, along with the running furrow version."""
    return {"status": "ok", "version": __version__}


@app.get("/api/config")
async def config() -> dict[str, str]:
    """Return the provider and effective model names; never any credentials."""
    return {
        "provider": _provider_name(settings.provider),
        "model": settings.model,
        "planner_model": settings.planner_model,
        "worker_model": settings.worker_model,
        "tester_model": settings.tester_model,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Run furrow for one goal, streaming every event back as a JSON line.

    The client sends ``{"goal": "...", "model": "..."}`` once; the run then
    streams events until it finishes, the client goes away, or the server shuts
    down. Disconnects stop the run cooperatively and never propagate.
    """
    run_task: asyncio.Task[RunResult] | None = None
    watcher: asyncio.Task[None] | None = None
    client: LLMClient | None = None
    await websocket.accept()
    try:
        try:
            request = StartRequest.model_validate(await websocket.receive_json())
        except WebSocketDisconnect:
            return
        except (ValueError, TypeError) as exc:
            await _send_event(websocket, make_event(EventKind.ERROR, f"invalid payload: {exc}"))
            return

        goal = request.goal.strip()
        if not goal:
            await _send_event(websocket, make_event(EventKind.ERROR, "goal must not be empty"))
            return

        cfg = Settings(**_overrides(request))
        client = LLMClient(settings=cfg)

        async def on_event(event: Event) -> None:
            await _send_event(websocket, event)

        orchestrator = Orchestrator(goal=goal, client=client, settings=cfg, on_event=on_event)
        run_task = asyncio.create_task(orchestrator.run())
        watcher = asyncio.create_task(_watch_disconnect(websocket))
        await asyncio.wait({run_task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if watcher.done() and not run_task.done():
            await orchestrator.stop()
            run_task.cancel()
        try:
            await run_task
        except WebSocketDisconnect:
            await orchestrator.stop()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await _send_event(websocket, make_event(EventKind.ERROR, f"run failed: {exc}"))
    finally:
        for task in (run_task, watcher):
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        if client is not None:
            with suppress(Exception):
                await client.aclose()
        await _close(websocket)


def run(host: str = "127.0.0.1", port: int = 8000, *, reload: bool = False) -> None:
    """Serve the furrow web app with uvicorn.

    Args:
        host: Interface to bind; loopback by default.
        port: Port to listen on.
        reload: Enable uvicorn's auto-reload (development only).

    Raises:
        ValueError: When ``port`` is outside the valid range.
    """
    if not 1 <= int(port) <= 65535:
        raise ValueError(f"port must be between 1 and 65535, got {port}")
    uvicorn.run(app, host=host, port=int(port), reload=reload)


def _overrides(request: StartRequest) -> dict[str, Any]:
    """Return ``Settings`` overrides for one request.

    Like the CLI's ``--model`` flag, a requested model applies to every role.
    """
    model = (request.model or "").strip()
    if not model:
        return {}
    return {
        "model": model,
        "planner_model": model,
        "worker_model": model,
        "tester_model": model,
    }


def _provider_name(provider: Any) -> str:
    """Return the string value of a provider setting."""
    return str(getattr(provider, "value", provider))


async def _send_event(websocket: WebSocket, event: Event) -> None:
    """Send one JSON-encoded event, tolerating a peer that already went away."""
    payload = json.dumps(event.to_dict())
    try:
        await websocket.send_text(payload)
    except (WebSocketDisconnect, RuntimeError, OSError):
        return


async def _close(websocket: WebSocket) -> None:
    """Close the socket so the browser sees ``onclose``, ignoring a dead peer."""
    with suppress(WebSocketDisconnect, RuntimeError, OSError):
        await websocket.close()


async def _watch_disconnect(websocket: WebSocket) -> None:
    """Block until the peer disconnects, consuming any further client frames."""
    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return