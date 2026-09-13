from __future__ import annotations

import hmac
import os

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from furrow.config import Settings, settings as global_settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

app = FastAPI(title="Furrow")


def _get_web_token() -> str | None:
    env_token = os.getenv("FURROW_WEB_TOKEN")
    if env_token:
        return env_token
    token_attr = getattr(global_settings, "web_token", None)
    if token_attr is not None:
        return token_attr.get_secret_value()
    return None


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
    <input id="token" placeholder="Token (optional)" />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\\nStarting...\\n';
      const token = document.getElementById('token').value;
      const wsProtocol = location.protocol === 'https:' ? 'wss://' : 'ws://';
      const wsUrl = wsProtocol + location.host + '/ws' + (token ? '?token=' + encodeURIComponent(token) : '');
      const ws = new WebSocket(wsUrl);
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
    token = _get_web_token()
    if token:
        query_token = websocket.query_params.get("token")
        if not query_token or not hmac.compare_digest(query_token, token):
            await websocket.close(code=4001, reason="Authentication required")
            return
    await websocket.accept()
    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        return

    goal = data.get("goal", "")
    model = data.get("model", None)

    if model:
        settings_obj = global_settings.model_copy(update={"model": model})
    else:
        settings_obj = global_settings
    client = LLMClient(settings=settings_obj)

    try:
        async def on_event(message: str) -> None:
            await websocket.send_text(message)

        orchestrator = Orchestrator(
            goal=goal,
            client=client,
            settings_obj=settings_obj,
            on_event=on_event,
        )
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    finally:
        await client.close()


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
