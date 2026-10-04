"""Offline tests for the web entrypoint and the CLI.

Everything here is fake: no LLM client is constructed for real, no network
traffic, no uvicorn server and no subprocesses.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, ClassVar

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from furrow.cli import main as cli_main
from furrow.config import settings
from furrow.web import server as web_server

STATE: dict[str, Any] = {
    "goal": "ship it",
    "original_goal": "ship it",
    "cycles": 1,
    "running": False,
    "stopped_reason": "goal_complete",
    "tasks": [
        {"id": "t1", "description": "one", "status": "completed"},
        {"id": "t2", "description": "two", "status": "completed"},
    ],
    "completed": 2,
    "failed": 0,
    "unfinished": 0,
}


class FakeOrchestrator:
    """Minimal stand-in for :class:`furrow.core.orchestrator.Orchestrator`."""

    instances: ClassVar[list[FakeOrchestrator]] = []

    # Test knobs, set on the class with monkeypatch.
    block: bool = False
    result: Any = STATE

    def __init__(
        self,
        goal: str,
        client: Any = None,
        *,
        on_event: Any = None,
        **kwargs: Any,
    ) -> None:
        self.goal = goal
        self.client = client
        self.on_event = on_event
        self.kwargs = kwargs
        self.stop_calls = 0
        self.finished = False
        self.state: dict[str, Any] = {"goal": goal, "cycles": 0, "running": True, "tasks": []}
        self._release = asyncio.Event()
        FakeOrchestrator.instances.append(self)

    def _emit(self, kind: str, **payload: Any) -> None:
        if self.on_event is not None:
            self.on_event({"kind": kind, "cycle": 0, **payload})

    def request_stop(self) -> None:
        self.stop_calls += 1
        self._release.set()

    @property
    def is_running(self) -> bool:
        return not self._release.is_set()

    async def run(self) -> Any:
        self._emit("run_start", goal=self.goal)
        self._emit("plan", tasks=[{"id": "t1", "description": "one"}])
        self._emit("task_start", task_id="t1", description="one")
        self._emit("test_result", passed=True, summary="1 passed")
        self._emit("task_end", task_id="t1", status="completed")
        if self.block:
            await self._release.wait()
        self.finished = True
        self.state = dict(self.result) if isinstance(self.result, dict) else {}
        self._emit("run_end", state=self.state)
        return self.result


class ExplodingRelay(web_server._EventRelay):
    """A relay whose callback raises after forwarding one event."""

    def __call__(self, event: dict[str, Any]) -> None:
        super().__call__(event)
        if event.get("kind") == "plan":
            raise RuntimeError("relay exploded")


class FakeLLMClient:
    """Captures the settings handed to it; never talks to a provider."""

    instances: ClassVar[list[FakeLLMClient]] = []

    def __init__(self, settings: Any = None, **kwargs: Any) -> None:
        self.settings = settings
        self.kwargs = kwargs
        FakeLLMClient.instances.append(self)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the server module state and the fakes clean between tests."""
    FakeOrchestrator.instances.clear()
    FakeLLMClient.instances.clear()
    monkeypatch.setattr(FakeOrchestrator, "block", False)
    monkeypatch.setattr(FakeOrchestrator, "result", STATE)
    monkeypatch.setattr(web_server, "_orchestrator_factory", FakeOrchestrator)
    monkeypatch.setattr(web_server, "_active_run", None)
    monkeypatch.setattr(web_server, "_last_state", None)
    monkeypatch.setattr(cli_main, "Orchestrator", FakeOrchestrator)
    monkeypatch.setattr(cli_main, "LLMClient", FakeLLMClient)


def _wait_until(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _collect(ws: Any, wanted: set[str], limit: int = 30) -> list[dict[str, Any]]:
    """Receive frames until one of ``wanted`` arrives (or ``limit`` is exceeded)."""
    frames: list[dict[str, Any]] = []
    for _ in range(limit):
        frame = ws.receive_json()
        frames.append(frame)
        if frame.get("kind") in wanted:
            return frames
    raise AssertionError(f"never saw {wanted}; got {[frame.get('kind') for frame in frames]}")


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------
def test_index_serves_goal_input_stop_button_and_relative_ws_url() -> None:
    with TestClient(web_server.app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    assert 'id="goal"' in html
    assert 'id="stop"' in html
    assert "Stop" in html
    assert "ws://" not in html
    assert "location.host" in html


def test_healthz_and_status_before_and_after_a_run() -> None:
    with TestClient(web_server.app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        before = client.get("/api/status")
        assert before.status_code == 200
        assert before.json() == {
            "running": False,
            "goal": None,
            "cycles": 0,
            "tasks": {"total": 0, "completed": 0, "failed": 0, "unfinished": 0},
            "stopped_reason": None,
        }

        with client.websocket_connect("/ws") as ws:
            ws.send_json({"goal": "ship it"})
            frames = _collect(ws, {"run_end"})

        after = client.get("/api/status").json()
        assert frames[-1]["kind"] == "run_end"
        assert after["running"] is False
        assert after["goal"] == "ship it"
        assert after["cycles"] == 1
        assert after["stopped_reason"] == "goal_complete"
        assert after["tasks"] == {"total": 2, "completed": 2, "failed": 0, "unfinished": 0}


def test_status_counts_mixed_task_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        FakeOrchestrator,
        "result",
        {
            "stopped_reason": "cycle_budget",
            "cycles": 3,
            "tasks": [
                {"id": "a", "status": "completed"},
                {"id": "b", "status": "failed"},
                {"id": "c", "status": "in_progress"},
                {"id": "d", "status": "pending"},
                "not-a-task",
            ],
        },
    )
    with TestClient(web_server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"goal": "mixed"})
            _collect(ws, {"run_end"})
        assert client.get("/api/status").json()["tasks"] == {
            "total": 4,
            "completed": 1,
            "failed": 1,
            "unfinished": 2,
        }


# ---------------------------------------------------------------------------
# Websocket
# ---------------------------------------------------------------------------
def test_ws_start_streams_events_then_reports_not_running() -> None:
    with TestClient(web_server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"goal": "ship it"})
            frames = _collect(ws, {"run_end"})

        kinds = [frame["kind"] for frame in frames]
        assert "run_start" in kinds
        assert "plan" in kinds
        assert "task_start" in kinds
        assert "test_result" in kinds
        assert "task_end" in kinds
        assert kinds[-1] == "run_end"

        assert _wait_until(lambda: client.get("/api/status").json()["running"] is False)
        assert client.get("/api/status").json()["running"] is False


def test_ws_stop_action_calls_request_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FakeOrchestrator, "block", True)
    with TestClient(web_server.app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "stop me"})
        _collect(ws, {"task_end"})
        ws.send_json({"action": "stop"})
        frames = _collect(ws, {"stop_acknowledged"})
    assert frames[-1]["kind"] == "stop_acknowledged"
    orch = FakeOrchestrator.instances[-1]
    assert orch.stop_calls >= 1
    assert _wait_until(lambda: orch.finished)


def test_ws_disconnect_stops_run_and_second_start_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(FakeOrchestrator, "block", True)
    with TestClient(web_server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"goal": "first goal"})
            _collect(ws, {"task_end"})
            assert web_server._status()["running"] is True

            # A second concurrent start is refused with a clear error frame.
            with client.websocket_connect("/ws") as other:
                other.send_json({"goal": "second goal"})
                frames = _collect(other, {"error"})
            assert "already in progress" in frames[-1]["message"]
            assert len(FakeOrchestrator.instances) == 1

        # Leaving the context disconnects the client, which must stop the run.
        orch = FakeOrchestrator.instances[-1]
        assert _wait_until(lambda: orch.stop_calls >= 1)
        assert _wait_until(lambda: orch.finished)
    assert web_server._status()["running"] is False


def test_ws_invalid_payloads_produce_error_frames() -> None:
    with TestClient(web_server.app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text("not json at all")
        assert "invalid JSON payload" in _collect(ws, {"error"})[-1]["message"]

        ws.send_json(["a", "list"])
        assert "invalid JSON payload" in _collect(ws, {"error"})[-1]["message"]

        ws.send_json({"goal": "   "})
        assert "goal must not be empty" in _collect(ws, {"error"})[-1]["message"]

        ws.send_json({"action": "stop"})
        assert "no run in progress" in _collect(ws, {"error"})[-1]["message"]

        ws.send_json({"action": "wat"})
        assert "unknown action" in _collect(ws, {"error"})[-1]["message"]

        ws.send_json({"nope": 1})
        assert "invalid start payload" in _collect(ws, {"error"})[-1]["message"]

        # The socket survived every error and still starts a run.
        ws.send_json({"goal": "ship it"})
        assert _collect(ws, {"run_end"})[-1]["kind"] == "run_end"


def test_ws_raising_event_callback_does_not_break_run_or_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_server, "_EventRelay", ExplodingRelay)
    with TestClient(web_server.app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "ship it"})
        frames = _collect(ws, {"run_end"})
        kinds = [frame["kind"] for frame in frames]
        assert kinds[-1] == "run_end"
        assert "plan" in kinds
        assert "task_start" in kinds
        # Still usable after the raising callback.
        ws.send_json({"action": "stop"})
        assert _collect(ws, {"stop_acknowledged"})[-1]["kind"] == "stop_acknowledged"
    assert FakeOrchestrator.instances[0].stop_calls >= 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_version_flag() -> None:
    command = CliRunner().invoke(cli_main.main, ["start", "--version"])
    assert command.exit_code == 0
    assert "furrow" in command.output.lower()

    group = CliRunner().invoke(cli_main.main, ["--version"])
    assert group.exit_code == 0
    assert "version" in group.output.lower()


def test_cli_start_forwards_options_without_mutating_global_settings() -> None:
    before = (settings.model, settings.max_cycles, settings.max_parallel_tasks)

    result = CliRunner().invoke(
        cli_main.main,
        ["start", "build it", "--model", "foo", "--max-cycles", "2", "--max-parallel", "3"],
    )
    assert result.exit_code == 0, result.output

    orch = FakeOrchestrator.instances[-1]
    assert orch.goal == "build it"
    assert isinstance(orch.client, FakeLLMClient)

    client = FakeLLMClient.instances[-1]
    assert client.settings.model == "foo"
    assert client.settings.max_cycles == 2
    assert client.settings.max_parallel_tasks == 3

    # The shared settings singleton must be untouched.
    assert (settings.model, settings.max_cycles, settings.max_parallel_tasks) == before


def test_cli_start_provider_option_is_forwarded() -> None:
    result = CliRunner().invoke(cli_main.main, ["start", "g", "--provider", "openai"])
    assert result.exit_code == 0, result.output
    assert FakeLLMClient.instances[-1].settings.provider.value == "openai"


def test_cli_start_exits_1_when_the_run_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        FakeOrchestrator,
        "result",
        {"stopped_reason": "cycle_budget", "cycles": 2, "failed": 1, "unfinished": 1},
    )
    result = CliRunner().invoke(cli_main.main, ["start", "nope"])
    assert result.exit_code == 1


def test_cli_start_keyboard_interrupt_exits_130(monkeypatch: pytest.MonkeyPatch) -> None:
    class InterruptingOrchestrator(FakeOrchestrator):
        async def run(self) -> Any:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli_main, "Orchestrator", InterruptingOrchestrator)
    result = CliRunner().invoke(cli_main.main, ["start", "stop me"])
    assert result.exit_code == 130
    assert "stopped by user" in result.output


def test_cli_web_passes_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []

    def fake_run(*args: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(web_server, "run", fake_run)
    result = CliRunner().invoke(cli_main.main, ["web", "--host", "127.0.0.1", "--port", "9999"])
    assert result.exit_code == 0, result.output
    assert calls == [{"host": "127.0.0.1", "port": 9999, "log_level": None}]