"""Tests for the Furrow web interface: the index page and the ``/ws`` stream."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("starlette")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from furrow.config import Settings  # noqa: E402
from furrow.web.server import app  # noqa: E402

#: (root handlers, root level, furrow handlers, furrow level) -- the logging
#: state the websocket endpoint snapshots before touching and restores after.
LoggingState = tuple[tuple[logging.Handler, ...], int, tuple[logging.Handler, ...], int]


def _snapshot_logging_state() -> LoggingState:
    """Capture the process-wide logging state the endpoint mutates."""
    root = logging.getLogger()
    furrow_logger = logging.getLogger("furrow")
    return (tuple(root.handlers), root.level, tuple(furrow_logger.handlers), furrow_logger.level)


class _OrchestratorSpy:
    """Records how the web layer drove the stubbed ``Orchestrator``."""

    def __init__(self) -> None:
        self.instances: list[Any] = []
        self.run_calls = 0
        self.run_side_effect: BaseException | None = None
        self.during_states: list[LoggingState] = []


@pytest.fixture(autouse=True)
def stub_orchestrator(monkeypatch: pytest.MonkeyPatch) -> _OrchestratorSpy:
    """Replace ``furrow.web.server.Orchestrator`` with a recording stub."""
    spy = _OrchestratorSpy()

    class _StubOrchestrator:
        def __init__(self, goal: str, **kwargs: Any) -> None:
            self.goal = goal
            self.kwargs = kwargs
            spy.instances.append(self)

        async def run(self) -> None:
            spy.run_calls += 1
            # Capture the "during" state for the no-leak test: the bridge
            # handler must be installed and both loggers lowered to INFO.
            spy.during_states.append(_snapshot_logging_state())
            if spy.run_side_effect is not None:
                raise spy.run_side_effect

    monkeypatch.setattr("furrow.web.server.Orchestrator", _StubOrchestrator)
    return spy


def _read_frames(
    payload: str, count: int, *, expect_disconnect: bool = False, timeout: float = 10.0
) -> list[Any]:
    """Send ``payload`` over a test websocket and return ``count`` frames.

    The read runs on a daemon thread and both context managers live inside
    it: starlette's ``WebSocketTestSession.__enter__`` is what assigns
    ``self.portal``, so a connection opened outside the thread fails with
    ``AttributeError: 'WebSocketTestSession' object has no attribute 'portal'``.
    """
    frames: list[Any] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            with TestClient(app) as client, client.websocket_connect("/ws") as ws:
                ws.send_text(payload)
                for _ in range(count):
                    frames.append(json.loads(ws.receive_text()))
                if expect_disconnect:
                    # receive_text() routes through _raise_on_close, so a
                    # server-side close surfaces as WebSocketDisconnect.
                    with pytest.raises(WebSocketDisconnect):
                        ws.receive_text()
        except BaseException as exc:  # noqa: B036 - re-raised on the main thread below
            errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        pytest.fail(f"websocket read did not finish within {timeout}s (likely a hang)")
    if errors:
        raise errors[0]
    return frames


def _wait_for_logging_state(expected: LoggingState, timeout: float = 2.0) -> None:
    """Poll until the logging state matches ``expected`` (bounded by ``timeout``).

    The endpoint restores its logging changes in its ``finally`` block, which
    runs on the portal's event-loop thread after the last frame is sent, so an
    immediate assertion would race the restore.
    """
    deadline = time.monotonic() + timeout
    while _snapshot_logging_state() != expected:
        if time.monotonic() >= deadline:
            pytest.fail(
                f"logging state was not restored within {timeout}s: "
                f"current={_snapshot_logging_state()!r} expected={expected!r}"
            )
        time.sleep(0.01)


def test_index_serves_ui_page() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert 'id="log"' in response.text


@pytest.mark.parametrize("payload", ['{"goal": ""}', "{}", '{"goal": "   "}'])
def test_blank_goal_yields_single_error_frame(payload: str) -> None:
    frames = _read_frames(payload, count=1)

    expected = {"type": "error", "message": "goal is required and must be non-empty"}
    assert frames == [expected]


def test_non_object_payload_reports_value_error() -> None:
    frames = _read_frames("[1, 2, 3]", count=1)

    assert frames == [{"type": "error", "message": "ValueError: expected a JSON object"}]


def test_successful_run_streams_accepted_cycle_start_done(
    stub_orchestrator: _OrchestratorSpy,
) -> None:
    frames = _read_frames(json.dumps({"goal": "ship the thing"}), count=3)

    assert frames == [
        {"type": "accepted", "goal": "ship the thing", "cycle": 0},
        {"type": "cycle_start", "cycle": 1},
        {"type": "done", "reason": "complete"},
    ]
    assert stub_orchestrator.run_calls == 1
    assert stub_orchestrator.instances[0].goal == "ship the thing"


def test_failing_run_streams_error_then_closes_cleanly(
    stub_orchestrator: _OrchestratorSpy,
) -> None:
    stub_orchestrator.run_side_effect = RuntimeError("kaboom")

    payload = json.dumps({"goal": "ship the thing"})
    frames = _read_frames(payload, count=3, expect_disconnect=True)

    assert [frame["type"] for frame in frames] == ["accepted", "cycle_start", "error"]
    assert frames[2] == {"type": "error", "message": "RuntimeError: kaboom"}
    assert stub_orchestrator.run_calls == 1


def test_model_option_reaches_orchestrator_settings(
    stub_orchestrator: _OrchestratorSpy,
) -> None:
    payload = json.dumps({"goal": "ship the thing", "model": "claude-opus-4"})
    _read_frames(payload, count=3)

    (orchestrator,) = stub_orchestrator.instances
    assert orchestrator.kwargs == {"settings": Settings(model="claude-opus-4")}


def test_logging_setup_is_restored_after_the_run(
    stub_orchestrator: _OrchestratorSpy,
) -> None:
    before = _snapshot_logging_state()
    payload = json.dumps({"goal": "ship the thing"})
    _read_frames(payload, count=3)

    assert stub_orchestrator.run_calls == 1
    (during,) = stub_orchestrator.during_states

    (before_root_handlers, _, before_furrow_handlers, _) = before
    (during_root_handlers, during_root_level, during_furrow_handlers, during_furrow_level) = during

    # While the run was active the bridge handler was appended to the root
    # logger and both loggers were lowered to INFO so progress reaches the
    # socket; the furrow logger itself gains no handlers.
    assert during_root_level == logging.INFO
    assert during_furrow_level == logging.INFO
    assert tuple(during_root_handlers[: len(before_root_handlers)]) == before_root_handlers
    assert len(during_root_handlers) == len(before_root_handlers) + 1
    assert during_furrow_handlers == before_furrow_handlers

    # The restore happens on the worker thread, so poll instead of asserting.
    _wait_for_logging_state(before)
