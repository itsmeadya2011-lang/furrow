from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

# Tags to strip from rich console output before sending to the browser.
_MARKUP_TAGS = (
    "[bold green]", "[/bold green]",
    "[bold cyan]", "[/bold cyan]",
    "[red]", "[/red]",
    "[yellow]", "[/yellow]",
    "[blue]", "[/blue]",
    "[bold]", "[/bold]",
)


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

    async def send_line(line: str) -> None:
        cleaned = line
        for tag in _MARKUP_TAGS:
            cleaned = cleaned.replace(tag, "")
        await websocket.send_text(cleaned + "\n")

    async def runner() -> None:
        try:
            data = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        goal = data.get("goal", "")
        if not goal:
            await send_line("No goal provided.")
            return
        orchestrator = Orchestrator(goal=goal, on_output=send_line)
        try:
            await orchestrator.run()
        except Exception as exc:  # pragma: no cover - defensive
            await send_line(f"[red]Error: {exc}[/red]")

    task = asyncio.create_task(runner())
    try:
        # Keep the connection open until the orchestrator finishes or the
        # client disconnects. Messages are pushed directly via on_output.
        while not task.done():
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                break
    finally:
        task.cancel()
        if not task.done():
            await task


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)