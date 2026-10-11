from __future__ import annotations

import asyncio
import json
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None
    provider: Optional[str] = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <input id="model" placeholder="model (optional)" />
    <select id="provider">
      <option value="">provider (default)</option>
      <option value="anthropic">anthropic</option>
      <option value="openai">openai</option>
      <option value="ollama">ollama</option>
    </select>
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');

    function escapeHtml(s) {
      return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }

    function appendLine(line) {
      const span = document.createElement('span');
      span.textContent = line;
      out.appendChild(span);
      out.appendChild(document.createTextNode('\\n'));
      out.scrollTop = out.scrollHeight;
    }

    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = '';
      const goal = document.getElementById('goal').value;
      const model = document.getElementById('model').value;
      const provider = document.getElementById('provider').value;
      const payload = {goal: goal};
      if (model) payload.model = model;
      if (provider) payload.provider = provider;

      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => ws.send(JSON.stringify(payload));
      ws.onmessage = (ev) => {
        let data;
        try { data = JSON.parse(ev.data); } catch (err) { appendLine('<<invalid message: ' + escapeHtml(ev.data) + '>>'); return; }
        switch (data.type) {
          case 'cycle':
            appendLine('── Cycle ' + data.cycle + ' ──');
            break;
          case 'plan':
            appendLine('Plan: ' + escapeHtml(data.plan.rationale));
            for (const t of data.plan.tasks) {
              appendLine('  [' + escapeHtml(t.id) + '] ' + escapeHtml(t.description));
            }
            break;
          case 'task':
            appendLine('Task ' + escapeHtml(data.task_id) + ': ' + escapeHtml(data.status));
            if (data.result) {
              appendLine('  ' + escapeHtml(data.result));
            }
            break;
          case 'test':
            if (data.passed) {
              appendLine('Tests passed: ' + escapeHtml(data.summary));
            } else {
              appendLine('Tests failed: ' + escapeHtml(data.summary));
              for (const f of data.failures) {
                appendLine('  ! ' + escapeHtml(f));
              }
            }
            break;
          case 'done':
            appendLine('Done: ' + escapeHtml(data.reason));
            break;
          default:
            appendLine('<<' + escapeHtml(JSON.stringify(data)) + '>>');
        }
      };
      ws.onclose = () => appendLine('Closed.');
      ws.onerror = () => appendLine('WebSocket error.');
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
        provider = data.get("provider")

        # Apply optional model/provider overrides to the shared settings,
        # consistent with the CLI behaviour.
        if model:
            settings.model = model
        if provider:
            settings.provider = Provider(provider)

        def on_event(event: dict) -> None:
            try:
                asyncio.get_running_loop().create_task(
                    websocket.send_text(json.dumps(event))
                )
            except Exception:
                pass

        orchestrator = Orchestrator(goal=goal, on_event=on_event)
        await orchestrator.run()

        # Ensure the client sees a clean close signal.
        try:
            await websocket.send_text(json.dumps({"type": "closed"}))
        except Exception:
            pass
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)