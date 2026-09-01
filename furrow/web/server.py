from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.core.session import SessionManager, generate_session_id

app = FastAPI(title="Furrow")
logger = logging.getLogger(__name__)

# Global state for the current websocket connection
_current_websocket: Optional[WebSocket] = None
_current_orchestrator: Optional[Orchestrator] = None


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class CreateSessionRequest(BaseModel):
    goal: str


def _get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _event_callback(event_type: str, data: dict[str, Any]) -> None:
    """Callback for orchestrator events - broadcasts to WebSocket."""
    message = {
        "type": event_type,
        "timestamp": _get_timestamp(),
        "data": data,
    }
    # Schedule the websocket send on the event loop
    if _current_websocket is not None:
        asyncio.create_task(_send_ws_message(message))


async def _send_ws_message(message: dict[str, Any]) -> None:
    """Send a message to the current WebSocket connection."""
    global _current_websocket
    if _current_websocket is not None:
        try:
            await _current_websocket.send_json(message)
        except Exception as exc:
            logger.warning("Failed to send WebSocket message: %s", exc)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=_DASHBOARD_HTML)


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": _get_timestamp(),
        "version": "0.1.0",
    }


@app.get("/api/sessions")
async def list_sessions() -> dict[str, Any]:
    """List all saved sessions."""
    session_manager = SessionManager(Settings().workspace)
    sessions = session_manager.list_sessions()
    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "goal": s.goal,
                "status": s.status,
                "cycles": s.cycles,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ],
        "count": len(sessions),
    }


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    """Get details of a specific session."""
    session_manager = SessionManager(Settings().workspace)
    try:
        state = session_manager.load(session_id)
        return {
            "session_id": state.session_id,
            "goal": state.goal,
            "current_goal": state.current_goal,
            "status": state.status,
            "cycles": state.cycles,
            "current_plan": state.current_plan,
            "workspace": state.workspace,
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
        }
    except Exception as exc:
        return {"error": str(exc)}, 404


@app.post("/api/sessions")
async def create_session(request: CreateSessionRequest) -> dict[str, Any]:
    """Create a new session without starting it."""
    session_manager = SessionManager(Settings().workspace)
    session_id = generate_session_id()
    sid, state = session_manager.new_session(goal=request.goal, session_id=session_id)
    return {
        "session_id": sid,
        "goal": state.goal,
        "status": state.status,
        "created_at": state.created_at.isoformat(),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    global _current_websocket, _current_orchestrator
    await websocket.accept()
    _current_websocket = websocket

    # Send initial state
    await websocket.send_json({
        "type": "connected",
        "timestamp": _get_timestamp(),
        "data": {
            "message": "Connected to Furrow",
            "status": "idle",
        },
    })

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        session_id = data.get("session_id")

        # Send status update
        await websocket.send_json({
            "type": "status",
            "timestamp": _get_timestamp(),
            "data": {
                "status": "starting",
                "goal": goal,
                "message": f"Starting orchestrator for goal: {goal}",
            },
        })

        # Create orchestrator with event callback
        orchestrator = Orchestrator(
            goal=goal,
            session_id=session_id,
            on_event=_event_callback,
        )
        _current_orchestrator = orchestrator

        # Send initial state
        await websocket.send_json({
            "type": "status",
            "timestamp": _get_timestamp(),
            "data": {
                "status": "running",
                "session_id": orchestrator.session_id,
                "goal": orchestrator.goal,
                "cycles": orchestrator.cycles,
            },
        })

        await orchestrator.run()

        # Send completion message
        await websocket.send_json({
            "type": "status",
            "timestamp": _get_timestamp(),
            "data": {
                "status": "completed",
                "session_id": orchestrator.session_id,
                "cycles": orchestrator.cycles,
                "message": "Orchestrator finished",
            },
        })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        try:
            await websocket.send_json({
                "type": "error",
                "timestamp": _get_timestamp(),
                "data": {
                    "error": str(exc),
                    "message": f"Error: {exc}",
                },
            })
        except Exception:
            pass
    finally:
        _current_websocket = None
        _current_orchestrator = None


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)


# Enhanced dashboard HTML with modern CSS
_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Furrow - AI Orchestration Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .pulse-dot { animation: pulse-dot 1.5s ease-in-out infinite; }
        
        @keyframes slide-in {
            from { opacity: 0; transform: translateY(-4px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .slide-in { animation: slide-in 0.2s ease-out; }
        
        .log-container {
            font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
        }
        
        .status-connecting { color: #f59e0b; }
        .status-running { color: #10b981; }
        .status-idle { color: #6b7280; }
        .status-completed { color: #3b82f6; }
        .status-error { color: #ef4444; }
        
        .task-pending { background: #f3f4f6; border-left: 3px solid #9ca3af; }
        .task-running { background: #fef3c7; border-left: 3px solid #f59e0b; }
        .task-completed { background: #d1fae5; border-left: 3px solid #10b981; }
        .task-failed { background: #fee2e2; border-left: 3px solid #ef4444; }
        
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #1f2937; }
        ::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #6b7280; }
    </style>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-6 max-w-6xl">
        <!-- Header -->
        <header class="mb-6">
            <div class="flex items-center justify-between">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 bg-gradient-to-br from-emerald-400 to-cyan-500 rounded-lg flex items-center justify-center">
                        <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                        </svg>
                    </div>
                    <div>
                        <h1 class="text-2xl font-bold text-white">Furrow</h1>
                        <p class="text-sm text-gray-400">AI Orchestration Dashboard</p>
                    </div>
                </div>
                <div id="connection-status" class="flex items-center gap-2 px-3 py-1.5 bg-gray-800 rounded-full">
                    <span class="w-2 h-2 rounded-full bg-gray-500 pulse-dot" id="status-dot"></span>
                    <span class="text-sm text-gray-400" id="status-text">Disconnected</span>
                </div>
            </div>
        </header>

        <!-- Main Input -->
        <div class="bg-gray-800 rounded-xl p-5 mb-6 shadow-lg">
            <form id="form" class="flex gap-3">
                <input 
                    id="goal" 
                    type="text"
                    placeholder="Enter your goal (e.g., 'Build a REST API for managing tasks')"
                    class="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                    required
                />
                <button 
                    type="submit"
                    id="submit-btn"
                    class="px-6 py-3 bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600 text-white font-medium rounded-lg transition-all duration-200 shadow-lg hover:shadow-emerald-500/25 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    Start
                </button>
            </form>
        </div>

        <!-- Stats Grid -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div class="bg-gray-800 rounded-xl p-4">
                <p class="text-xs text-gray-400 uppercase tracking-wide">Session ID</p>
                <p id="session-id" class="text-sm font-mono text-gray-300 mt-1 truncate">-</p>
            </div>
            <div class="bg-gray-800 rounded-xl p-4">
                <p class="text-xs text-gray-400 uppercase tracking-wide">Cycles</p>
                <p id="cycles-count" class="text-2xl font-bold text-white mt-1">0</p>
            </div>
            <div class="bg-gray-800 rounded-xl p-4">
                <p class="text-xs text-gray-400 uppercase tracking-wide">Tasks</p>
                <p id="tasks-count" class="text-2xl font-bold text-white mt-1">0/0</p>
            </div>
            <div class="bg-gray-800 rounded-xl p-4">
                <p class="text-xs text-gray-400 uppercase tracking-wide">Current Phase</p>
                <p id="current-phase" class="text-sm font-medium text-emerald-400 mt-1">Idle</p>
            </div>
        </div>

        <!-- Progress Bar -->
        <div class="bg-gray-800 rounded-xl p-4 mb-6">
            <div class="flex justify-between items-center mb-2">
                <span class="text-sm text-gray-400">Progress</span>
                <span id="progress-text" class="text-sm text-gray-400">0%</span>
            </div>
            <div class="w-full bg-gray-700 rounded-full h-2.5 overflow-hidden">
                <div id="progress-bar" class="bg-gradient-to-r from-emerald-500 to-cyan-500 h-2.5 rounded-full transition-all duration-500" style="width: 0%"></div>
            </div>
        </div>

        <!-- Main Content Grid -->
        <div class="grid md:grid-cols-2 gap-6">
            <!-- Task List -->
            <div class="bg-gray-800 rounded-xl p-5">
                <h2 class="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <svg class="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
                    </svg>
                    Tasks
                </h2>
                <div id="task-list" class="space-y-2 max-h-80 overflow-y-auto">
                    <p class="text-gray-500 text-sm text-center py-8">No tasks yet. Enter a goal to begin.</p>
                </div>
            </div>

            <!-- Log Output -->
            <div class="bg-gray-800 rounded-xl p-5">
                <h2 class="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <svg class="w-5 h-5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"/>
                    </svg>
                    Activity Log
                </h2>
                <div id="log-output" class="log-container bg-gray-900 rounded-lg p-4 h-80 overflow-y-auto space-y-1">
                    <p class="text-gray-500">Waiting for connection...</p>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer class="mt-8 text-center text-gray-500 text-sm">
            <p>Furrow v0.1.0 - AI-Powered Task Orchestration</p>
        </footer>
    </div>

    <script>
        // State
        let ws = null;
        let tasks = [];
        let currentStatus = 'idle';
        let cycles = 0;

        // DOM Elements
        const form = document.getElementById('form');
        const goalInput = document.getElementById('goal');
        const submitBtn = document.getElementById('submit-btn');
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');
        const sessionIdEl = document.getElementById('session-id');
        const cyclesEl = document.getElementById('cycles-count');
        const tasksEl = document.getElementById('tasks-count');
        const currentPhaseEl = document.getElementById('current-phase');
        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-text');
        const taskList = document.getElementById('task-list');
        const logOutput = document.getElementById('log-output');

        // Utility Functions
        function formatTime(timestamp) {
            const date = new Date(timestamp);
            return date.toLocaleTimeString('en-US', { hour12: false });
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function updateStatus(status) {
            currentStatus = status;
            const statusClasses = {
                'connecting': { dot: 'bg-yellow-500', text: 'Connecting...', textClass: 'text-yellow-400' },
                'idle': { dot: 'bg-gray-500', text: 'Idle', textClass: 'text-gray-400' },
                'running': { dot: 'bg-emerald-500', text: 'Running', textClass: 'text-emerald-400' },
                'starting': { dot: 'bg-yellow-500', text: 'Starting...', textClass: 'text-yellow-400' },
                'completed': { dot: 'bg-blue-500', text: 'Completed', textClass: 'text-blue-400' },
                'error': { dot: 'bg-red-500', text: 'Error', textClass: 'text-red-400' },
                'paused': { dot: 'bg-orange-500', text: 'Paused', textClass: 'text-orange-400' },
            };
            const config = statusClasses[status] || statusClasses['idle'];
            statusDot.className = `w-2 h-2 rounded-full ${config.dot} pulse-dot`;
            statusText.textContent = config.text;
            statusText.className = `text-sm ${config.textClass}`;
        }

        function addLog(message, type = 'info') {
            const colors = {
                'info': 'text-gray-300',
                'success': 'text-emerald-400',
                'error': 'text-red-400',
                'warning': 'text-yellow-400',
                'system': 'text-cyan-400',
            };
            const color = colors[type] || colors['info'];
            const time = formatTime(new Date().toISOString());
            const p = document.createElement('p');
            p.className = `slide-in ${color}`;
            p.innerHTML = `<span class="text-gray-500">[${time}]</span> ${escapeHtml(message)}`;
            logOutput.appendChild(p);
            logOutput.scrollTop = logOutput.scrollHeight;
        }

        function updateProgress() {
            if (tasks.length === 0) {
                progressBar.style.width = '0%';
                progressText.textContent = '0%';
                return;
            }
            const completed = tasks.filter(t => t.status === 'completed').length;
            const percent = Math.round((completed / tasks.length) * 100);
            progressBar.style.width = `${percent}%`;
            progressText.textContent = `${percent}%`;
            tasksEl.textContent = `${completed}/${tasks.length}`;
        }

        function renderTasks() {
            if (tasks.length === 0) {
                taskList.innerHTML = '<p class="text-gray-500 text-sm text-center py-8">No tasks yet. Enter a goal to begin.</p>';
                return;
            }
            taskList.innerHTML = tasks.map(task => {
                const statusClass = `task-${task.status}`;
                const statusIcon = {
                    'pending': '<span class="text-gray-400">○</span>',
                    'running': '<span class="text-yellow-400 pulse-dot">●</span>',
                    'completed': '<span class="text-emerald-400">✓</span>',
                    'failed': '<span class="text-red-400">✗</span>',
                }[task.status] || '○';
                return `
                    <div class="${statusClass} rounded-lg p-3 slide-in">
                        <div class="flex items-start gap-2">
                            <span class="mt-0.5">${statusIcon}</span>
                            <div class="flex-1 min-w-0">
                                <p class="text-sm font-medium text-gray-200 truncate">${escapeHtml(task.description)}</p>
                                <p class="text-xs text-gray-400 mt-0.5">ID: ${escapeHtml(task.id)}</p>
                                ${task.result ? `<p class="text-xs text-gray-400 mt-1 truncate">${escapeHtml(task.result)}</p>` : ''}
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        function updateTasks(newTasks) {
            tasks = newTasks;
            renderTasks();
            updateProgress();
        }

        // WebSocket Connection
        function connect(goal) {
            updateStatus('connecting');
            addLog('Connecting to server...', 'system');
            
            ws = new WebSocket(`ws://${location.host}/ws`);
            
            ws.onopen = () => {
                addLog('Connected to Furrow server', 'system');
                ws.send(JSON.stringify({ goal: goal }));
            };

            ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    handleMessage(msg);
                } catch (e) {
                    addLog(`Raw: ${event.data}`, 'info');
                }
            };

            ws.onclose = () => {
                updateStatus('idle');
                addLog('Disconnected from server', 'system');
                submitBtn.disabled = false;
                submitBtn.textContent = 'Start';
            };

            ws.onerror = (error) => {
                updateStatus('error');
                addLog('WebSocket error occurred', 'error');
            };
        }

        function handleMessage(msg) {
            const { type, timestamp, data } = msg;
            const time = formatTime(timestamp);

            switch (type) {
                case 'connected':
                    updateStatus('idle');
                    addLog(data.message, 'system');
                    break;

                case 'status':
                    if (data.status) updateStatus(data.status);
                    if (data.session_id) sessionIdEl.textContent = data.session_id;
                    if (data.cycles !== undefined) {
                        cycles = data.cycles;
                        cyclesEl.textContent = cycles;
                    }
                    if (data.message) addLog(data.message, data.status === 'error' ? 'error' : 'info');
                    break;

                case 'log':
                    const logType = data.passed === false ? 'error' : (data.phase === 'testing' ? 'warning' : 'info');
                    addLog(data.message, logType);
                    if (data.phase) currentPhaseEl.textContent = data.phase.charAt(0).toUpperCase() + data.phase.slice(1);
                    break;

                case 'cycle_start':
                    currentPhaseEl.textContent = 'Cycle ' + data.cycle;
                    addLog(data.message, 'system');
                    break;

                case 'cycle_complete':
                    if (data.tasks) updateTasks(data.tasks);
                    addLog(data.message, 'success');
                    break;

                case 'plan_created':
                    addLog(`Plan created with ${data.tasks.length} tasks`, 'info');
                    break;

                case 'task_update':
                    if (data.tasks) updateTasks(data.tasks);
                    if (data.task_id && data.status) {
                        const taskStatus = data.status === 'completed' ? 'success' : (data.status === 'failed' ? 'error' : 'info');
                        addLog(`Task ${data.task_id}: ${data.status}`, taskStatus);
                    }
                    break;

                case 'error':
                    updateStatus('error');
                    addLog(data.message || data.error, 'error');
                    break;

                default:
                    addLog(`${type}: ${JSON.stringify(data)}`, 'info');
            }
        }

        // Form Handler
        form.onsubmit = (e) => {
            e.preventDefault();
            const goal = goalInput.value.trim();
            if (!goal) return;
            
            submitBtn.disabled = true;
            submitBtn.textContent = 'Running...';
            logOutput.innerHTML = '';
            tasks = [];
            cycles = 0;
            cyclesEl.textContent = '0';
            sessionIdEl.textContent = '-';
            currentPhaseEl.textContent = 'Starting';
            
            connect(goal);
        };

        // Initial status
        updateStatus('idle');
    </script>
</body>
</html>
"""
