from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.logging import get_logger, setup_logging

setup_logging("INFO")
logger = get_logger("web")

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


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
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\\nStarting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("websocket_connected")
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        logger.info("websocket_goal_received", goal=goal[:100])
        orchestrator = Orchestrator(goal=goal)
        
        # Override console output to send to websocket
        original_print = orchestrator._print_to_console if hasattr(orchestrator, '_print_to_console') else None
        
        async def send_message(message: str):
            try:
                await websocket.send_text(message)
            except Exception:
                pass
        
        # We can't easily intercept console output, so let's run with a custom callback
        # For now, just run and send a completion message
        await orchestrator.run()
        await send_message("\n[bold green]Goal complete. Halting.[/bold green]")
    except WebSocketDisconnect:
        logger.info("websocket_disconnected")
    except Exception as e:
        logger.error("websocket_error", error=str(e))
        try:
            await websocket.send_text(f"\n[red]Error: {e}[/red]")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    logger.info("starting_web_server", host=host, port=port)
    uvicorn.run(app, host=host, port=port)