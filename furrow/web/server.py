from __future__ import annotations

import hmac
import ipaddress
import os
from typing import Optional
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    model: Optional[str] = None
    token: Optional[str] = None

    @field_validator("goal")
    @classmethod
    def goal_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Goal must not be blank")
        return value


def _request_error(payload: dict[object, object]) -> str:
    unknown = sorted(str(key) for key in payload if key not in {"goal", "model", "token"})
    if unknown:
        noun = "field" if len(unknown) == 1 else "fields"
        return f"Unknown {noun}: {', '.join(unknown)}"
    if "goal" not in payload:
        return "Goal is required"
    goal = payload["goal"]
    if not isinstance(goal, str):
        return "Goal must be a string"
    if not goal.strip():
        return "Goal must not be blank"
    return "Invalid request"


async def _reject(websocket: WebSocket, message: str) -> None:
    try:
        await websocket.send_json({"error": message})
    except (RuntimeError, WebSocketDisconnect):
        pass
    try:
        await websocket.close(code=1008, reason=message)
    except (RuntimeError, WebSocketDisconnect):
        pass


async def _close(websocket: WebSocket, code: int = 1000) -> None:
    try:
        await websocket.close(code=code)
    except (RuntimeError, WebSocketDisconnect):
        pass


def _is_local_origin(origin: str | None) -> bool:
    if origin is None:
        return True
    parsed = urlparse(origin.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return False
    hostname = parsed.hostname.lower()
    if hostname in {"localhost", "localhost."}:
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _is_authenticated(websocket: WebSocket, payload: dict[object, object]) -> bool:
    configured = os.environ.get("FURROW_WEB_TOKEN")
    if not configured:
        return True

    candidates: list[str | None] = [
        websocket.query_params.get("token"),
        websocket.headers.get("x-furrow-token"),
    ]
    authorization = websocket.headers.get("authorization")
    if authorization:
        scheme, separator, value = authorization.partition(" ")
        candidates.append(value if separator and scheme.lower() == "bearer" else authorization)

    payload_token = payload.get("token")
    if isinstance(payload_token, str):
        candidates.append(payload_token)

    return any(
        isinstance(candidate, str) and hmac.compare_digest(candidate, configured)
        for candidate in candidates
    )


def _build_orchestrator(request: StartRequest) -> Orchestrator:
    client: LLMClient | None = None
    if request.model is not None:
        request_settings: Settings = settings.model_copy(update={"model": request.model})
        client = LLMClient(settings=request_settings)
    return Orchestrator(goal=request.goal, client=client)


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
    <input
      id="token"
      type="password"
      placeholder="Web token (optional)"
      autocomplete="current-password"
    />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = (e) => {
      e.preventDefault();
      out.textContent += '\\nStarting...\\n';
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(protocol + '//' + location.host + '/ws');
      ws.onopen = () => {
        const payload = {goal: document.getElementById('goal').value};
        const token = document.getElementById('token').value;
        if (token) payload.token = token;
        ws.send(JSON.stringify(payload));
      };
      ws.onmessage = (ev) => {
        let message = null;
        try { message = JSON.parse(ev.data); } catch (error) {}
        if (message && message.error) {
          out.textContent += 'Error: ' + message.error + '\\n';
        } else {
          out.textContent += ev.data + '\\n';
        }
      };
      ws.onerror = () => out.textContent += '\\nConnection error.\\n';
      ws.onclose = (event) => {
        const code = event.code ? ' (' + event.code + ')' : '';
        out.textContent += '\\nClosed' + code + '.\\n';
      };
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    try:
        await websocket.accept()
    except (RuntimeError, WebSocketDisconnect):
        return

    try:
        try:
            payload = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        except (ValueError, RuntimeError):
            await _reject(websocket, "Invalid JSON")
            return

        if not isinstance(payload, dict):
            await _reject(websocket, "Payload must be a JSON object")
            return

        try:
            request = StartRequest.model_validate(payload)
        except ValidationError as exc:
            await _reject(websocket, _request_error(payload))
            return

        token_configured = bool(os.environ.get("FURROW_WEB_TOKEN"))
        if token_configured and not _is_local_origin(websocket.headers.get("origin")):
            await _reject(websocket, "Origin not allowed")
            return
        if token_configured and not _is_authenticated(websocket, payload):
            await _reject(websocket, "Unauthorized")
            return

        orchestrator = _build_orchestrator(request)
        try:
            await orchestrator.run()
        except WebSocketDisconnect:
            pass
        finally:
            await _close(websocket)
    except WebSocketDisconnect:
        pass


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
