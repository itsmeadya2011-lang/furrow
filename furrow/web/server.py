from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from furrow.core.orchestrator import Orchestrator
from furrow.core.state import load_state

app = FastAPI(title="Furrow")

# Currently running orchestrator, if any (single-user local tool).
_current: Orchestrator | None = None


INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <meta charset="utf-8" />
  <style>
    body {
      background-color: #111;
      color: #eee;
      font-family: monospace;
      margin: 0;
      padding: 2rem;
    }
    h1 {
      margin-top: 0;
    }
    #log {
      background-color: #000;
      color: #0f0;
      padding: 1rem;
      height: 400px;
      overflow-y: scroll;
      white-space: pre-wrap;
      border: 1px solid #333;
    }
    #status {
      margin-top: 1rem;
      color: #ffa500;
    }
    input,
    button {
      font-family: monospace;
    }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button id="start" type="submit">Start</button>
  </form>
  <pre id="log"></pre>
  <div id="status"></div>
  <script>
    const logEl = document.getElementById("log");
    const statusEl = document.getElementById("status");
    const form = document.getElementById("form");
    const goalInput = document.getElementById("goal");
    const startButton = document.getElementById("start");

    const log = (msg) => {
      logEl.textContent += msg + "\\n";
      logEl.scrollTop = logEl.scrollHeight;
    };

    const setStatus = (msg) => {
      statusEl.textContent = msg;
    };

    const setFormEnabled = (enabled) => {
      startButton.disabled = !enabled;
      goalInput.disabled = !enabled;
    };

    const render = (event) => {
      switch (event.event) {
        case "start":
          log("Goal: " + event.goal);
          break;
        case "cycle":
          log("── Cycle " + event.cycle + " ──");
          break;
        case "plan":
          log(
            "Planned " +
              event.tasks.length +
              " task(s) — " +
              event.rationale
          );
          break;
        case "task_start":
          log("  ▶ task " + event.task_id + " started");
          break;
        case "task_result":
          log("  ✔ task " + event.task_id + " " + event.status);
          break;
        case "tests":
          log(
            "Tests " +
              (event.passed ? "PASSED" : "FAILED") +
              ": " +
              event.summary
          );
          if (event.failures && event.failures.length > 0) {
            for (let i = 0; i < event.failures.length; i++) {
              log("    ✗ " + event.failures[i]);
            }
          }
          break;
        case "cycle_end":
          log("Cycle " + event.cycle + " complete: " + event.done);
          break;
        case "done":
          log(
            "Finished after " +
              event.cycles +
              " cycle(s) — " +
              event.reason
          );
          setStatus("");
          setFormEnabled(true);
          break;
        case "error":
          log("ERROR: " + event.error);
          break;
        default:
          break;
      }
    };

    form.onsubmit = (e) => {
      e.preventDefault();
      setFormEnabled(false);
      setStatus("Running...");
      logEl.textContent = "";
      const ws = new WebSocket("ws://" + location.host + "/ws");
      ws.onopen = () => {
        ws.send(JSON.stringify({ goal: goalInput.value }));
      };
      ws.onmessage = (ev) => {
        try {
          render(JSON.parse(ev.data));
        } catch (err) {
          log("WebSocket message error: " + err);
        }
      };
      ws.onclose = () => {
        setFormEnabled(true);
        setStatus("Disconnected");
      };
      ws.onerror = () => {
        log("WebSocket error");
      };
    };
  </script>
</body>
</html>"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/state")
async def get_state() -> dict[str, Any]:
    if _current is not None:
        return _current.state()
    return load_state() or {"status": "idle"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    global _current
    await websocket.accept()
    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        return
    except Exception:
        await _safe_send(
            websocket,
            {"event": "error", "error": "Invalid request: expected JSON with a 'goal' field"},
        )
        return

    goal = (data or {}).get("goal")
    if not isinstance(goal, str) or not goal.strip():
        await _safe_send(
            websocket,
            {"event": "error", "error": "A non-empty 'goal' string is required"},
        )
        return

    async def send(event: dict) -> None:
        await _safe_send(websocket, event)

    orchestrator = Orchestrator(goal=goal, on_event=send)
    _current = orchestrator
    try:
        await orchestrator.run()
    except Exception as e:  # run() normally handles errors, but be safe
        await _safe_send(websocket, {"event": "error", "error": str(e)})
    finally:
        _current = None


async def _safe_send(websocket: WebSocket, event: dict) -> None:
    try:
        await websocket.send_json(event)
    except Exception:
        pass  # client disconnected; orchestrator keeps running


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)