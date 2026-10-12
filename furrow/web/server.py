from __future__ import annotations

from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 1rem; }
    #events { font-family: monospace; white-space: pre-wrap; margin: 1rem 0; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; min-height: 200px; }
    .cyan { color: #00ffff; }
    .blue { color: #60a0ff; }
    .green { color: #4ade80; }
    .red { color: #f87171; }
    .yellow { color: #fde047; }
    .bold { font-weight: bold; }
    button:disabled { opacity: 0.5; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit" id="start">Start</button>
    <button type="button" id="clear">Clear</button>
  </form>
  <div id="events"></div>
  <script>
    const form = document.getElementById('form');
    const goalInput = document.getElementById('goal');
    const startBtn = document.getElementById('start');
    const clearBtn = document.getElementById('clear');
    const events = document.getElementById('events');

    function escapeHtml(s) {
      return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function colorFor(type) {
      switch (type) {
        case 'start': case 'cycle_start': return 'cyan';
        case 'plan': return 'blue';
        case 'task_complete': return 'green';
        case 'task_failed': return 'red';
        case 'test_result': return 'green';
        case 'done': return 'green bold';
        case 'max_cycles_reached': return 'yellow';
        default: return '';
      }
    }

    function formatEvent(ev) {
      const type = ev.type;
      let color = colorFor(type);
      let text = '';
      switch (type) {
        case 'start':
          text = '[start] goal: ' + (ev.goal || '');
          break;
        case 'cycle_start':
          text = '[cycle_start] cycle ' + ev.cycle;
          break;
        case 'plan': {
          const tasks = (ev.plan && ev.plan.tasks) ? ev.plan.tasks : [];
          text = '[plan] cycle ' + ev.cycle + '\\n' +
                 tasks.map(t => '  - ' + t.id + ': ' + t.description).join('\\n');
          break;
        }
        case 'task_complete': {
          const t = ev.task || {};
          text = '[task_complete] ' + (t.id || '') + ': ' + (t.result || '');
          break;
        }
        case 'task_failed': {
          const t = ev.task || {};
          text = '[task_failed] ' + (t.id || '') + ': ' + (ev.error || '');
          break;
        }
        case 'test_result': {
          const r = ev.result || {};
          color = r.passed ? 'green' : 'red';
          text = '[test_result] cycle ' + ev.cycle + ' passed=' + r.passed + ' summary=' + (r.summary || '');
          if (r.failures && r.failures.length) {
            text += '\\n' + r.failures.map(f => '  ! ' + f).join('\\n');
          }
          break;
        }
        case 'done':
          text = '[done] cycles ' + ev.cycles;
          break;
        case 'max_cycles_reached':
          text = '[max_cycles_reached] cycles ' + ev.cycles;
          break;
        default:
          text = JSON.stringify(ev);
      }
      const cls = color;
      const lines = text.split('\\n').map(l => '<span class="' + cls + '">' + escapeHtml(l) + '</span>');
      return '<div style="white-space:pre-wrap">' + lines.join('<br>') + '</div>';
    }

    function addEvent(ev) {
      events.insertAdjacentHTML('beforeend', formatEvent(ev));
      events.scrollTop = events.scrollHeight;
    }

    function setRunning(running) {
      startBtn.disabled = running;
      goalInput.readOnly = running;
    }

    clearBtn.onclick = function () { events.innerHTML = ''; };

    form.onsubmit = async (e) => {
      e.preventDefault();
      events.innerHTML = '';
      setRunning(true);
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = function () {
        ws.send(JSON.stringify({ goal: goalInput.value }));
      };
      ws.onmessage = function (ev) {
        try {
          addEvent(JSON.parse(ev.data));
        } catch (err) {
          addEvent({ type: 'error', raw: ev.data });
        }
      };
      ws.onerror = function () { setRunning(false); };
      ws.onclose = function () { setRunning(false); };
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        orchestrator = Orchestrator(goal=goal)
        if model:
            orchestrator.settings.model = model

        async def on_event(event: dict) -> None:
            try:
                await websocket.send_json(event)
            except Exception:
                pass

        orchestrator.on_event.append(on_event)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)