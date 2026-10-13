from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from furrow.core.events import EventBus
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <meta charset="utf-8">
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 24px; background: #0d1117; color: #e6edf3; max-width: 900px; margin: 0 auto; }
    h1 { font-size: 1.75rem; margin-bottom: 16px; }
    .status { padding: 8px 12px; border-radius: 6px; margin-bottom: 16px; font-size: 0.875rem; font-weight: 600; display: inline-block; }
    .status.connecting { background: #238636; }
    .status.running { background: #0969da; }
    .status.closed { background: #8b949e; }
    .status.error { background: #da3633; }
    .form-row { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
    #goal { flex: 1; min-width: 300px; padding: 10px 12px; border-radius: 6px; border: 1px solid #30363d; background: #161b22; color: #e6edf3; font-size: 1rem; }
    #goal:focus { outline: none; border-color: #58a6ff; }
    button[type=submit] { padding: 10px 20px; border-radius: 6px; border: none; background: #238636; color: white; font-size: 1rem; font-weight: 600; cursor: pointer; }
    button[type=submit]:disabled { background: #30363d; color: #8b949e; cursor: not-allowed; }
    button[type=submit]:hover:not(:disabled) { background: #2ea043; }
    #log { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; height: 500px; overflow-y: auto; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.8rem; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word; }
    .event { margin-bottom: 8px; padding: 8px 10px; border-radius: 4px; background: #1c2128; border: 1px solid #30363d; }
    .event-header { display: flex; gap: 8px; align-items: center; margin-bottom: 4px; font-size: 0.75rem; }
    .badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; }
    .badge-plan { background: #1f6feb; color: white; }
    .badge-task { background: #a371f7; color: white; }
    .badge-test { background: #f78166; color: white; }
    .badge-halt { background: #238636; color: white; }
    .badge-error { background: #da3633; color: white; }
    .badge-session { background: #8b949e; color: white; }
    .badge-default { background: #30363d; color: #e6edf3; }
    .event-time { color: #8b949e; font-family: monospace; }
    .event-payload { color: #c9d1d9; font-size: 0.75rem; margin-left: 4px; }
    .event-payload.string { color: #e6edf3; }
    .event-payload.object { color: #c9d1d9; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <div id="status" class="status closed">Closed</div>
  <form id="form" class="form-row">
    <input id="goal" type="text" placeholder="Enter goal" required autocomplete="off" />
    <button type="submit">Start</button>
  </form>
  <div id="log"></div>
  <script>
    const form = document.getElementById('form');
    const goalInput = document.getElementById('goal');
    const statusEl = document.getElementById('status');
    const logEl = document.getElementById('log');
    let ws = null;

    function setStatus(text, cls) {
      statusEl.textContent = text;
      statusEl.className = 'status ' + cls;
    }

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    function formatPayload(payload) {
      if (payload === null || payload === undefined) {
        return '<span class="event-payload string">(empty)</span>';
      }
      if (typeof payload === 'string') {
        return '<span class="event-payload string">' + escapeHtml(payload) + '</span>';
      }
      try {
        const json = JSON.stringify(payload, null, 2);
        return '<span class="event-payload object">' + escapeHtml(json) + '</span>';
      } catch {
        return '<span class="event-payload string">' + escapeHtml(String(payload)) + '</span>';
      }
    }

    function badgeClass(type) {
      if (!type) return 'badge-default';
      const t = type.toLowerCase();
      if (t.includes('plan')) return 'badge-plan';
      if (t.includes('task')) return 'badge-task';
      if (t.includes('test')) return 'badge-test';
      if (t === 'halt') return 'badge-halt';
      if (t === 'error') return 'badge-error';
      if (t.startsWith('session')) return 'badge-session';
      return 'badge-default';
    }

    function appendEvent(event) {
      const div = document.createElement('div');
      div.className = 'event';
      const time = new Date().toLocaleTimeString();
      const type = event.type || 'unknown';
      const badge = '<span class="badge ' + badgeClass(type) + '">' + escapeHtml(type) + '</span>';
      const payloadHtml = formatPayload(event.payload ?? event);
      div.innerHTML = '<div class="event-header"><span class="event-time">' + time + '</span>' + badge + '</div>' + payloadHtml;
      logEl.appendChild(div);
      logEl.scrollTop = logEl.scrollHeight;
    }

    function enableForm(enabled) {
      goalInput.disabled = !enabled;
      form.querySelector('button[type=submit]').disabled = !enabled;
    }

    async function connect() {
      if (ws) { ws.close(); }
      setStatus('Connecting...', 'connecting');
      enableForm(false);
      logEl.innerHTML = '';

      ws = new WebSocket('ws://' + location.host + '/ws');

      ws.onopen = () => {
        setStatus('Running', 'running');
        ws.send(JSON.stringify({ goal: goalInput.value, model: null }));
      };

      ws.onmessage = (ev) => {
        try {
          const event = JSON.parse(ev.data);
          appendEvent(event);
          if (event.type === 'session.end' || event.type === 'halt' || event.type === 'error') {
            setStatus('Closed', 'closed');
            enableForm(true);
            ws.close();
          }
        } catch (e) {
          appendEvent({ type: 'error', payload: 'Failed to parse message: ' + e });
        }
      };

      ws.onerror = (err) => {
        appendEvent({ type: 'error', payload: 'WebSocket error' });
      };

      ws.onclose = () => {
        if (statusEl.classList.contains('running') || statusEl.classList.contains('connecting')) {
          setStatus('Closed', 'closed');
          enableForm(true);
        }
      };
    }

    form.addEventListener('submit', (e) => {
      e.preventDefault();
      if (!goalInput.value.trim()) return;
      connect();
    });
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue[dict] = asyncio.Queue()
    drain_task: asyncio.Task | None = None

    async def drain_queue() -> None:
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event)
                queue.task_done()
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001, S110
            pass

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")

        bus = EventBus()

        def forwarder(event: dict) -> None:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

        bus.subscribe(forwarder)

        orchestrator = Orchestrator(goal=goal, event_bus=bus)
        if model is not None:
            orchestrator.client.settings.model = model

        drain_task = asyncio.create_task(drain_queue())

        await orchestrator.run()

        await websocket.send_json({"type": "session.end"})

    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:  # noqa: BLE001, S110
            pass
    finally:
        if drain_task is not None:
            drain_task.cancel()
            try:
                await drain_task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001, S110
                pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)