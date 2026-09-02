from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import StateManager

_active_connections: list[WebSocket] = []
_sessions: dict[str, StateManager] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events."""
    yield
    # On shutdown, persist all session state
    for sm in _sessions.values():
        sm.save()


app = FastAPI(title="Furrow", lifespan=lifespan)


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None
    max_cycles: Optional[int] = None


class StatusResponse(BaseModel):
    session_id: Optional[str] = None
    goal: Optional[str] = None
    status: Optional[str] = None
    cycle: int = 0
    tasks: list[dict] = Field(default_factory=list)
    test_history: list[dict] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: monospace; margin: 2em; max-width: 900px; }
    pre { background: #1e1e1e; color: #d4d4d4; padding: 1em; border-radius: 6px; overflow-x: auto; }
    input, button { padding: 0.5em; font-size: 1em; margin: 0.2em 0; }
    button { background: #4CAF50; color: white; border: none; cursor: pointer; }
    button:hover { background: #45a049; }
    h1 { color: #333; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required style="width: 80%;" />
    <input id="maxCycles" type="number" placeholder="Max cycles (0 = unlimited)" />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      const maxCycles = document.getElementById('maxCycles').value;
      out.textContent += '\\nStarting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.onopen = () => {
        ws.send(JSON.stringify({goal: goal, maxCycles: maxCycles ? parseInt(maxCycles) : undefined}));
      };
    };
  </script>
</body>
</html>
"""
    )


@app.get("/status")
async def get_status() -> StatusResponse:
    """Return the status of the most recent or active session."""
    if not _sessions:
        return StatusResponse(cycle=0)
    # Get the most recently active session
    session_id = next(reversed(_sessions))
    sm = _sessions[session_id]
    state = sm.state
    return StatusResponse(
        session_id=session_id,
        goal=state.goal,
        status=state.status.value,
        cycle=state.cycle,
        tasks=[t.model_dump() for t in state.tasks],
        test_history=state.test_history,
        errors=state.errors,
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    if len(_active_connections) >= 10:
        await websocket.close(code=1013)  # try again later
        return
    await websocket.accept()
    _active_connections.append(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        if not goal:
            await websocket.send_text("Error: No goal provided.")
            return

        max_cycles = data.get("maxCycles", 0) or 0

        settings = Settings()
        settings.max_cycles = max_cycles

        orchestrator = Orchestrator(
            goal=goal,
            settings=settings,
        )
        _sessions[goal] = orchestrator.state_manager

        # Stream output to the client
        async def run_with_output():
            try:
                state = await orchestrator.run()
                await websocket.send_text(
                    f"\n[SESSION_COMPLETE] status={state.status.value}"
                )
            except Exception as e:
                await websocket.send_text(f"\n[ERROR] {e}")

        await run_with_output()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _active_connections:
            _active_connections.remove(websocket)
        # Clean up this session's tracking — state already persisted
        goal_var = locals().get("goal")
        if goal_var and goal_var in _sessions:
            _sessions[goal_var].save()
            del _sessions[goal_var]


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the Furrow web UI server."""
    import asyncio

    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        pass
