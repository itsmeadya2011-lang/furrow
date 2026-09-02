from __future__ import annotations

from typing import Optional

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Provider, Settings, settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

log = structlog.get_logger(__name__)

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None
    provider: Optional[str] = None


INDEX_HTML = """
<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <input id="model" placeholder="Model override (optional)" />
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
      ws.send(JSON.stringify({
        goal: document.getElementById('goal').value,
        model: document.getElementById('model').value || null,
      }));
    };
  </script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


def _build_ws_settings(data: dict[str, object]) -> Settings:
    """Create a Settings instance honouring optional model/provider overrides."""
    updates: dict[str, object] = {}
    model_override = data.get("model")
    if model_override:
        updates["model"] = model_override
        updates["worker_model"] = model_override
    provider_override = data.get("provider")
    if provider_override:
        updates["provider"] = Provider(provider_override)
    return settings.model_copy(update=updates)  # type: ignore[arg-type]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    async def ws_callback(text: str) -> None:
        await websocket.send_text(text)

    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        await log.ainfo("web.disconnect.before_start")
        return

    goal = data.get("goal", "")
    ws_settings = _build_ws_settings(data)
    llm_client = LLMClient(settings=ws_settings)

    orchestrator = Orchestrator(
        goal=goal,
        client=llm_client,
        settings=ws_settings,
        on_output=ws_callback,
    )
    try:
        await orchestrator.run()
    except WebSocketDisconnect:
        await log.ainfo("web.disconnect.during_run")
    except Exception as e:
        await log.aerror("web.error", error=str(e))
        try:
            await websocket.send_text(f"[red]Error: {e}[/red]")
        except WebSocketDisconnect:
            pass
    finally:
        await llm_client.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
