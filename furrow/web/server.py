from __future__ import annotations

import json

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class BroadcastHub:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    def add(self, websocket: WebSocket) -> None:
        self._connections.add(websocket)

    def remove(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    async def broadcast(self, message: dict) -> None:
        if not self._connections:
            return
        data = json.dumps(message)
        stale: list[WebSocket] = []
        for connection in list(self._connections):
            try:
                await connection.send_text(data)
            except (WebSocketDisconnect, RuntimeError, OSError, ValueError):
                stale.append(connection)
        for connection in stale:
            self.remove(connection)


hub = BroadcastHub()

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Furrow</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background:#0f172a;color:#e2e8f0;line-height:1.6;padding:20px}.container{max-width:1200px;margin:0 auto}header{margin-bottom:24px}h1{color:#10b981;font-size:28px}.card{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:16px;margin-bottom:16px}.card h2{font-size:16px;margin-bottom:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}.form-group{display:flex;gap:12px}input[type=text]{flex:1;padding:10px 12px;background:#0f172a;border:1px solid #475569;border-radius:6px;color:#e2e8f0;font-size:14px}input[type=text]:focus{outline:none;border-color:#10b981}button{padding:10px 20px;background:#10b981;color:#0f172a;border:none;border-radius:6px;font-size:14px;font-weight:600;cursor:pointer}button:hover{background:#059669}button:disabled{background:#475569;cursor:not-allowed}.status-row{display:flex;gap:16px;align-items:center}.status-badge{padding:6px 12px;border-radius:6px;font-size:13px;font-weight:600}.status-idle{background:#334155;color:#94a3b8}.status-running{background:#164e63;color:#22d3ee;animation:pulse 2s ease-in-out infinite}@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}.cycle-counter{color:#94a3b8;font-size:14px}.goal-text{color:#e2e8f0;font-size:14px;word-break:break-word}table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;padding:8px 12px;background:#0f172a;color:#94a3b8;border:1px solid #334155;font-size:12px;text-transform:uppercase}td{padding:8px 12px;border:1px solid #334155;color:#cbd5e1}.task-pending{color:#94a3b8}.task-running{color:#fbbf24}.task-completed{color:#10b981}.task-failed{color:#f87171}.status-cell{font-weight:600}.test-status{padding:8px 12px;border-radius:6px;font-size:13px;margin-bottom:8px}.test-passed{background:#064e3b;color:#10b981}.test-failed{background:#7f1d1d;color:#f87171}.test-summary{color:#cbd5e1;font-size:13px;margin-bottom:8px}.failures-list{list-style:none}.failures-list li{background:#1e293b;padding:6px 10px;margin-bottom:4px;border-left:3px solid #f87171;color:#fca5a5;font-size:12px}.log-area{background:#0f172a;border:1px solid #334155;border-radius:6px;padding:12px;max-height:400px;overflow-y:auto;font-family:monospace;font-size:11px;color:#94a3b8;white-space:pre-wrap}.log-entry{margin-bottom:4px}
</style>
</head>
<body>
<div class="container">
<header><h1>Furrow</h1></header>
<div class="card"><h2>Start Goal</h2><form id="form" class="form-group"><input type="text" id="goal" placeholder="Enter goal..." required autocomplete="off"/><button type="submit" id="startBtn">Start</button></form></div>
<div class="card"><h2>Status</h2><div class="status-row"><div id="statusBadge" class="status-badge status-idle">Idle</div><div class="cycle-counter">Cycle: <span id="cycleCounter">0</span></div></div></div>
<div class="card"><h2>Goal</h2><div id="goalDisplay" class="goal-text">-</div></div>
<div class="card"><h2>Tasks</h2><div style="overflow-x:auto"><table><thead><tr><th>ID</th><th>Description</th><th>Status</th><th>Result</th></tr></thead><tbody id="tasksBody"></tbody></table></div></div>
<div class="card"><h2>Test Results</h2><div id="testPanel"></div></div>
<div class="card"><h2>Log</h2><div id="logArea" class="log-area"></div></div>
</div>
<script>
const form=document.getElementById("form"),goalInput=document.getElementById("goal"),startBtn=document.getElementById("startBtn"),statusBadge=document.getElementById("statusBadge"),cycleCounter=document.getElementById("cycleCounter"),goalDisplay=document.getElementById("goalDisplay"),tasksBody=document.getElementById("tasksBody"),testPanel=document.getElementById("testPanel"),logArea=document.getElementById("logArea");let ws=null,tasks=[];
function log(m){const e=document.createElement("div");e.className="log-entry";e.textContent="["+new Date().toLocaleTimeString()+"] "+m;logArea.appendChild(e);logArea.scrollTop=logArea.scrollHeight;}
function setStatus(s){if(s==="running"){statusBadge.textContent="Running";statusBadge.className="status-badge status-running";startBtn.disabled=true;}else{statusBadge.textContent="Idle";statusBadge.className="status-badge status-idle";startBtn.disabled=false;}}
function updateCycle(c){cycleCounter.textContent=c;}
function escapeHtml(t){if(t===null||t===undefined)return"";const d=document.createElement("div");d.textContent=t;return d.innerHTML;}
function renderTasks(){if(tasks.length===0){tasksBody.innerHTML='<tr><td colspan="4" style="text-align:center;color:#64748b;">No tasks yet</td></tr>';return;}tasksBody.innerHTML="";for(const task of tasks){const tr=document.createElement("tr");const sc="task-"+(task.status||"pending");tr.innerHTML="<td>"+escapeHtml(task.id)+"</td><td>"+escapeHtml(task.description||"")+"</td><td class="status-cell "+sc+"">"+escapeHtml(task.status||"pending")+"</td><td>"+escapeHtml(task.result||"")+"</td>";tasksBody.appendChild(tr);}}
form.onsubmit=e=>{e.preventDefault();const goal=goalInput.value.trim();if(!goal)return;tasks=[];testPanel.innerHTML="";logArea.innerHTML="";renderTasks();goalDisplay.textContent=goal;setStatus("running");updateCycle(0);log("Connecting...");ws=new WebSocket("ws://"+location.host+"/ws");ws.onopen=()=>{log("Connected. Starting...");ws.send(JSON.stringify({goal:goal}));};ws.onmessage=ev=>{try{handle(JSON.parse(ev.data));}catch(err){log("Parse error");}};ws.onclose=()=>{log("Closed");setStatus("idle");};ws.onerror=()=>{log("Error");setStatus("idle");}};
function handle(msg){log("Message: "+msg.type);if(msg.type==="cycle_start"){updateCycle(msg.cycle||0);}else if(msg.type==="plan"){tasks=(msg.tasks||[]).map(t=>({id:t.id,description:t.description,status:t.status||"pending",result:t.result||""}));renderTasks();}else if(msg.type==="task_complete"){const ex=tasks.find(t=>t.id===msg.task_id);if(ex){ex.status=msg.status;ex.result=msg.result;}else{tasks.push({id:msg.task_id,description:"",status:msg.status,result:msg.result});}renderTasks();}else if(msg.type==="test_result"){testPanel.innerHTML="";const d=document.createElement("div");d.className="test-status "+(msg.passed?"test-passed":"test-failed");d.textContent=msg.passed?"PASSED":"FAILED";testPanel.appendChild(d);const s=document.createElement("div");s.className="test-summary";s.textContent=msg.summary||"";testPanel.appendChild(s);if(msg.failures&&msg.failures.length>0){const ul=document.createElement("ul");ul.className="failures-list";for(const f of msg.failures){const li=document.createElement("li");li.textContent=f;ul.appendChild(li);}testPanel.appendChild(ul);}}else if(msg.type==="goal_complete"||msg.type==="max_cycles_reached"||msg.type==="done"){setStatus("idle");log(msg.type);}}
</script>
</body>
</html>"""


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=HTML_TEMPLATE)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    hub.add(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal, message_callback=hub.broadcast)
        await orchestrator.run()
        await websocket.send_text(json.dumps({"type": "done"}))
    except WebSocketDisconnect:
        hub.remove(websocket)
    except (RuntimeError, ValueError, OSError):
        hub.remove(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
