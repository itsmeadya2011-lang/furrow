from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient
from furrow.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("ws_connected", client_count=len(self.active_connections))
    
    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("ws_disconnected", client_count=len(self.active_connections))
    
    async def broadcast(self, message: str) -> None:
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except (OSError, RuntimeError) as e:
                logger.debug("broadcast_failed", error=str(e))
                # Connection might be closed


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("web_server_started")
    yield
    # Shutdown
    logger.info("web_server_stopped")


app = FastAPI(title="Furrow", lifespan=lifespan)


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class ConsoleCapture:
    """Capture console output and broadcast to WebSocket."""
    
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self._original_print = None
    
    def __enter__(self):
        import sys
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import sys
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
    
    def write(self, text: str) -> int:
        if text.strip():
            asyncio.create_task(self.websocket.send_text(text))
        return len(text)
    
    def flush(self) -> None:
        pass


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
        #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; 
               min-height: 400px; max-height: 600px; overflow-y: auto; white-space: pre-wrap; }
        input { width: 70%; padding: 0.5rem; font-size: 1rem; }
        button { padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
        .status { color: #4ec9b0; }
        .error { color: #f44747; }
        .warning { color: #cca700; }
        .success { color: #4ec9b0; }
    </style>
</head>
<body>
    <h1>Furrow</h1>
    <form id="form">
        <input id="goal" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
        <button type="submit">Start</button>
    </form>
    <pre id="out"></pre>
    <script>
        const form = document.getElementById('form');
        const out = document.getElementById('out');
        let ws = null;
        
        form.onsubmit = async (e) => {
            e.preventDefault();
            const goal = document.getElementById('goal').value;
            out.textContent = '';
            
            if (ws) {
                ws.close();
            }
            
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onopen = () => {
                ws.send(JSON.stringify({goal: goal}));
            };
            ws.onmessage = (ev) => {
                const line = ev.data;
                const span = document.createElement('div');
                if (line.includes('error') || line.includes('failed') || line.includes('Error')) {
                    span.className = 'error';
                } else if (line.includes('warning') || line.includes('Warning')) {
                    span.className = 'warning';
                } else if (line.includes('passed') || line.includes('completed') || line.includes('complete')) {
                    span.className = 'success';
                } else if (line.includes('Planning') || line.includes('Executing') || line.includes('Testing')) {
                    span.className = 'status';
                }
                span.textContent = line;
                out.appendChild(span);
                out.scrollTop = out.scrollHeight;
            };
            ws.onclose = () => {
                const span = document.createElement('div');
                span.className = 'status';
                span.textContent = '\\n[Connection closed]';
                out.appendChild(span);
            };
        };
    </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        
        # Create orchestrator with custom settings if model provided
        settings = Settings()
        if model:
            settings.model = model
        
        client = LLMClient(settings=settings)
        orchestrator = Orchestrator(goal=goal, client=client, orchestrator_settings=settings)
        
        # Override console print to capture output
        from rich.console import Console
        _ = Console()  # Initialize to ensure it's available
        
        class WebSocketConsole:
            def __init__(self, ws: WebSocket):
                self.ws = ws
            
            def print(self, *args, **kwargs):
                from io import StringIO
                buffer = StringIO()
                console = Console(file=buffer, force_terminal=True)
                console.print(*args, **kwargs)
                output = buffer.getvalue()
                if output.strip():
                    asyncio.create_task(self.ws.send_text(output))
        
        ws_console = WebSocketConsole(websocket)
        
        # Monkey-patch the console used by orchestrator
        import furrow.core.orchestrator as orch_module
        original_console_obj = orch_module.console
        orch_module.console = ws_console  # type: ignore[assignment]
        
        try:
            await orchestrator.run()
        finally:
            # Restore original console
            orch_module.console = original_console_obj
            
    except WebSocketDisconnect:
        pass
    except (OSError, RuntimeError) as e:
        logger.error("ws_error", error=str(e))
        try:
            await websocket.send_text(f"Error: {e}")
        except (OSError, RuntimeError) as send_error:
            logger.debug("ws_send_error_failed", error=str(send_error))
    finally:
        manager.disconnect(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
