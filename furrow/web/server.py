from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

# Store active orchestrator for feedback
active_orchestrator: Orchestrator | None = None


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class FeedbackRequest(BaseModel):
    feedback: str


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
  <div id="feedback-form" style="display:none; margin-top:10px;">
    <input id="feedback" placeholder="Enter feedback" />
    <button type="button" onclick="sendFeedback()">Send Feedback</button>
    <button type="button" onclick="pauseOrchestrator()">Pause</button>
    <button type="button" onclick="resumeOrchestrator()">Resume</button>
  </div>
  <pre id="out"></pre>
  <script>
    let ws = null;
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const feedbackForm = document.getElementById('feedback-form');
    
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\\nStarting...\\n';
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => {
        out.textContent += '\\nClosed.\\n';
        feedbackForm.style.display = 'none';
      };
      ws.onopen = () => {
        feedbackForm.style.display = 'block';
      };
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
    
    function sendFeedback() {
      const feedback = document.getElementById('feedback').value;
      if (feedback && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({feedback: feedback}));
        document.getElementById('feedback').value = '';
      }
    }
    
    function pauseOrchestrator() {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({action: 'pause'}));
      }
    }
    
    function resumeOrchestrator() {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({action: 'resume'}));
      }
    }
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    global active_orchestrator
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        def on_output(message: str) -> None:
            asyncio.create_task(websocket.send_text(message))

        orchestrator = Orchestrator(goal=goal, on_output=on_output)
        active_orchestrator = orchestrator
        
        # Start orchestrator in background
        run_task = asyncio.create_task(orchestrator.run())
        
        # Handle incoming messages (feedback, pause, resume)
        while not run_task.done():
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.5)
                if "feedback" in msg:
                    orchestrator.add_feedback(msg["feedback"])
                elif msg.get("action") == "pause":
                    orchestrator.pause()
                elif msg.get("action") == "resume":
                    orchestrator.resume()
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                break
        
        await run_task
        await websocket.send_text("\n[bold green]Done.[/bold green]")
        active_orchestrator = None
    except WebSocketDisconnect:
        active_orchestrator = None
        pass
    except Exception as e:
        await websocket.send_text(f"\n[red]Error: {e}[/red]")
        active_orchestrator = None


@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest) -> dict:
    """HTTP endpoint for sending feedback to the active orchestrator."""
    if active_orchestrator:
        active_orchestrator.add_feedback(request.feedback)
        return {"status": "ok"}
    return {"status": "no_active_orchestrator"}


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)