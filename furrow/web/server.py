from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from furrow.config import Provider, Settings, settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

logger = logging.getLogger("furrow.web")

app = FastAPI(title="Furrow")

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# How long we wait for queued events to reach the client before giving up.
SEND_DRAIN_TIMEOUT = 5.0

# ``on_event`` signature: Callable[[str, dict[str, Any]], None]
EventCallback = Callable[[str, dict[str, Any]], None]


def _event_payload(event: str, **data: Any) -> str:
    """Serialise an event for the browser. ``default=str`` keeps odd values printable."""
    return json.dumps({"event": event, "data": data}, default=str)


class _EventBridge:
    """Pushes orchestrator events to a websocket.

    ``on_event`` is a plain synchronous callback and may be called from a worker
    thread, so payloads are handed to the loop with ``call_soon_threadsafe`` and a
    separate sender task performs the actual ``send_text``.
    """

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.closed = False

    async def send(self, event: str, **data: Any) -> None:
        """Send one event immediately (used for acknowledgements and errors)."""
        await self._ws.send_text(_event_payload(event, **data))

    def push(self, event: str, data: dict[str, Any]) -> None:
        """Queue an event. Safe to call from any thread; never blocks."""
        if self.closed:
            return
        self._post(_event_payload(event, **data))

    def log(self, text: str) -> None:
        self.push("log", {"text": text})

    def finish(self) -> None:
        """Ask :meth:`pump` to drain and return."""
        self._post(None)

    async def pump(self) -> None:
        """Drain queued events until :meth:`finish` or the client goes away."""
        while True:
            item = await self._queue.get()
            if item is None:
                return
            try:
                await self._ws.send_text(item)
            except (WebSocketDisconnect, RuntimeError) as exc:
                # Client vanished mid-run: stop streaming but let the run finish.
                self.closed = True
                logger.info("websocket send failed, dropping events: %s", exc)
                return

    def _post(self, item: str | None) -> None:
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:  # loop already closed (server shutting down)
            pass


class _StdoutTee:
    """File-like shim mirroring writes to the real stdout and to a callback."""

    def __init__(self, stream: Any, emit: Callable[[str], None]) -> None:
        self._stream = stream
        self._emit = emit
        self._pending = ""
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        try:
            self._stream.write(text)
        except (OSError, ValueError) as exc:
            # Closed pipe / detached stream: never let output break the run.
            logger.debug("stdout tee write failed: %s", exc)
        with self._lock:
            self._pending += text
            *lines, self._pending = self._pending.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped:
                with contextlib.suppress(Exception):
                    self._emit(stripped)
        return len(text)

    def flush(self) -> None:
        with contextlib.suppress(Exception):
            self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        # Delegate isatty/encoding/fileno/... to the real stream.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._stream, name)


@contextlib.contextmanager
def _stdout_tee(emit: Callable[[str], None]) -> Iterator[None]:
    """Mirror printed output to the client.

    Fallback for orchestrator versions without the ``on_event`` hook: they report
    progress by printing, so stream what they print.
    """
    original = sys.stdout
    if original is None:
        yield
        return
    sys.stdout = _StdoutTee(original, emit)
    try:
        yield
    finally:
        sys.stdout = original


def _settings_for(request: dict[str, Any]) -> Settings:
    """Per-run settings, so one request's overrides don't leak into the next."""
    updates: dict[str, Any] = {}
    model = request.get("model")
    if isinstance(model, str) and model.strip():
        updates["model"] = model.strip()
    provider = request.get("provider")
    if isinstance(provider, str) and provider.strip():
        try:
            updates["provider"] = Provider(provider.strip().lower())
        except ValueError:
            logger.warning("Ignoring unknown provider %r", provider)
    if not updates:
        return settings
    return settings.model_copy(update=updates)


def _orchestrator_kwargs(callback: EventCallback, run_settings: Settings) -> dict[str, Any]:
    """Build Orchestrator kwargs, passing only what the installed class accepts.

    ``settings`` and ``on_event`` are newer constructor parameters, so inspecting
    the signature keeps this endpoint working against older orchestrators.
    """
    kwargs: dict[str, Any] = {}
    parameters = inspect.signature(Orchestrator).parameters
    if "client" in parameters:
        kwargs["client"] = LLMClient(settings=run_settings)
    if "settings" in parameters:
        kwargs["settings"] = run_settings
    if "on_event" in parameters:
        kwargs["on_event"] = callback
    return kwargs


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
    const goal = document.getElementById('goal');
    const out = document.getElementById('out');
    // Follow the page scheme so wss:// is used behind TLS-terminating proxies.
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    form.onsubmit = (e) => {
      e.preventDefault();
      const text = goal.value.trim();
      if (!text) { out.textContent += '\\nEnter a goal first.\\n'; return; }
      out.textContent += '\\nStarting...\\n';
      const ws = new WebSocket(scheme + '://' + location.host + '/ws');
      ws.onopen = () => ws.send(JSON.stringify({goal: text}));
      ws.onmessage = (ev) => {
        let line = ev.data;
        try {
          const msg = JSON.parse(ev.data);
          const data = msg.data || {};
          line = '[' + msg.event + '] ' + (data.message || data.text || JSON.stringify(data));
        } catch (err) { /* not JSON: show it raw */ }
        out.textContent += line + '\\n';
      };
      ws.onerror = () => out.textContent += '\\nWebSocket error.\\n';
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    bridge = _EventBridge(websocket)
    try:
        try:
            request = await websocket.receive_json()
        except WebSocketDisconnect:
            raise
        except (AssertionError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            # A malformed frame, a non-JSON body, or a socket that is no longer
            # connected: report it and close rather than leaking a traceback.
            logger.info("invalid websocket request: %s", exc)
            await bridge.send("error", message=f"Invalid request: {exc}")
            await websocket.close(code=1003)
            return
        if not isinstance(request, dict):
            await bridge.send("error", message="Expected a JSON object with a 'goal' field.")
            await websocket.close(code=1003)
            return

        goal = str(request.get("goal") or "").strip()
        if not goal:
            await bridge.send("error", message="A goal is required.")
            await websocket.close(code=1008)
            return

        # Acknowledge immediately so a long run never looks like a hang.
        await bridge.send("start", goal=goal, message="Run started.")

        run_settings = _settings_for(request)
        try:
            kwargs = _orchestrator_kwargs(bridge.push, run_settings)
            orchestrator = Orchestrator(goal=goal, **kwargs)
        except Exception as exc:
            logger.exception("could not start orchestrator")
            await bridge.send("error", message=f"Could not start run: {exc}")
            await websocket.close(code=1011)
            return

        # Older orchestrators have no on_event hook; stream their printed output.
        streamer = contextlib.nullcontext() if "on_event" in kwargs else _stdout_tee(bridge.log)
        pump = asyncio.create_task(bridge.pump(), name="furrow-events")
        try:
            with streamer:
                try:
                    await orchestrator.run()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # A bad API key or an unparseable plan lands here: report it
                    # to the client instead of tearing down the process.
                    logger.exception("orchestrator run failed")
                    await bridge.send("error", message=f"{type(exc).__name__}: {exc}")
                finally:
                    with contextlib.suppress(Exception):
                        await bridge.send("done", message="Run finished.")
                    bridge.finish()
        finally:
            try:
                await asyncio.wait_for(pump, timeout=SEND_DRAIN_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("timed out flushing events to client")
                pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump
    except WebSocketDisconnect:
        logger.info("websocket client disconnected")
    except Exception as exc:
        logger.exception("websocket endpoint failed")
        with contextlib.suppress(Exception):
            await bridge.send("error", message=f"{type(exc).__name__}: {exc}")
    finally:
        bridge.finish()
        with contextlib.suppress(Exception):
            await websocket.close()


def _env_port() -> int | None:
    raw = os.getenv("FURROW_PORT")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid FURROW_PORT=%r", raw)
        return None


def run(host: str | None = None, port: int | None = None) -> None:
    """Serve the UI. Defaults: $FURROW_HOST / 0.0.0.0 and $FURROW_PORT / 8000."""
    resolved_host = host or os.getenv("FURROW_HOST") or DEFAULT_HOST
    resolved_port = port if port is not None else _env_port()
    if resolved_port is None:
        resolved_port = DEFAULT_PORT
    uvicorn.run(app, host=resolved_host, port=resolved_port)
