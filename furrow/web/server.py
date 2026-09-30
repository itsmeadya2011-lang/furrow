from __future__ import annotations

import asyncio
import io

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

app = FastAPI(title="Furrow")
HTML_CONTENT = """
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
"""


class _WebSocketOutput(io.StringIO):
    """StringIO-based file-like object that forwards console output to a WebSocket in real-time."""

    def __init__(self, send_func) -> None:
        super().__init__()
        self._send_func = send_func
        self._pending = ""

    def write(self, s: str) -> int:
        result = super().write(s)
        self._pending += s
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            if line.strip():
                asyncio.ensure_future(self._send_func(line))
        return result

    def flush(self) -> None:
        if self._pending.strip():
            asyncio.ensure_future(self._send_func(self._pending))
        self._pending = ""
        self.seek(0)
        self.truncate(0)
        super().flush()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=HTML_CONTENT)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    async def _send(text: str) -> None:
        try:
            await websocket.send_text(text)
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001,S110
            pass

    import furrow.core.orchestrator as orch_module

    ws_output = _WebSocketOutput(send_func=_send)
    original_console = orch_module.console
    orch_module.console = Console(
        file=ws_output,
        force_terminal=False,
        width=120,
    )

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")

        client = None
        if model:
            client = LLMClient(settings=Settings(model=model))

        orchestrator = Orchestrator(goal=goal, client=client)
        await orchestrator.run()
        await _send("[Run complete]")
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        await _send(f"[Error: {e}]")
    finally:
        orch_module.console = original_console
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001,S110
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
