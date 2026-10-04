"""Tests for :mod:`furrow.web.server`: the HTTP surface and the WebSocket stream.

``Orchestrator.run`` is replaced throughout, so no loop is ever driven and nothing
is written to disk. The WebSocket tests drive the real ASGI app through
``fastapi.testclient.TestClient``, which exercises the queue bridge, the
multi-message loop and the disconnect handler.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import furrow.web.server as web_module
from furrow.core.events import EventType, FurrowEvent, RunResult
from furrow.web.server import ErrorFrame, EventFrame, RunEndFrame, StartRequest


@pytest.fixture
def client() -> Iterator[Any]:
    """Return a ``TestClient`` bound to the Furrow app.

    Yields:
        A context-managed test client.
    """
    with TestClient(web_module.app) as test_client:
        yield test_client


def _event(
    event_type: EventType, message: str = "", task_id: str | None = None, **data: Any
) -> FurrowEvent:
    """Build a :class:`FurrowEvent`.

    Args:
        event_type: Which lifecycle step.
        message: The human-readable summary.
        task_id: The task the event concerns.
        **data: Extra payload.

    Returns:
        The event.
    """
    return FurrowEvent(type=event_type, message=message, cycle=1, task_id=task_id, data=data)


def _install_run(
    monkeypatch: pytest.MonkeyPatch,
    events: list[FurrowEvent] | None = None,
    result: RunResult | None = None,
    error: BaseException | None = None,
) -> list[str]:
    """Replace ``Orchestrator.run`` with a scripted coroutine.

    The replacement drives the *real* ``Orchestrator._emit``, so the queue bridge
    under test is the production one.

    Args:
        monkeypatch: Pytest's patcher.
        events: Events published through ``on_event`` before the run finishes.
        result: The :class:`RunResult` to return. Defaults to a completed run.
        error: An exception to raise instead of returning, when set.

    Returns:
        A list recording the goal each fake orchestrator was built for, which
        also proves the socket accepted more than one message.
    """
    goals: list[str] = []
    final = (
        result
        if result is not None
        else RunResult(cycles=1, completed=True, tasks_completed=1, stopped_reason="all done")
    )

    async def fake_run(self: Any) -> RunResult:
        """Emit the scripted events and return the scripted result.

        Args:
            self: The orchestrator instance the endpoint built.

        Returns:
            ``final``.

        Raises:
            BaseException: ``error`` when one was scripted.
        """
        goals.append(self.goal)
        self.cycles += 1
        for event in events or []:
            self._emit(event.type, message=event.message, task_id=event.task_id, **event.data)
        if error is not None:
            raise error
        return final

    monkeypatch.setattr(web_module.Orchestrator, "run", fake_run)
    return goals


#: Frame types after which the server has nothing more to say about a run.
TERMINAL_FRAMES = frozenset({"run_end", "error"})


def _receive_or_none(socket: Any) -> Any:
    """Receive one raw ASGI message, or ``None`` if the socket is finished.

    Args:
        socket: A live ``WebSocketTestSession``.

    Returns:
        The raw message, or ``None`` when the server closed the socket.
    """
    try:
        raw = socket.receive()
    except WebSocketDisconnect:
        return None
    if raw["type"] == "websocket.close":
        return None
    return raw


def _collect_ws(client: Any, messages: list[Any]) -> list[dict[str, Any]]:
    """Drive one socket through `messages`, sending all goals upfront.

    All client payloads are sent back-to-back before any frame is read. The
    server's always-on reader task buffers every inbound frame onto an
    unbounded queue the instant it arrives, so runs execute serially in the
    order the goals were queued. This function then reads frames until it has
    collected a terminal frame (``run_end`` or ``error``) for each message sent.

    Each receive is bounded by a timeout so the helper can never hang the suite;
    a timeout produces a clear failure message naming what was still missing.

    Args:
        client: An open ``TestClient``.
        messages: Client payloads to send, in order.

    Returns:
        Every received frame, in order.

    Raises:
        AssertionError: If the expected number of terminal frames is not received
            within the timeout.
    """
    import time

    frames: list[dict[str, Any]] = []
    expected_terminal = len(messages)
    timeout_per_frame = 5.0

    with client.websocket_connect("/ws") as socket:
        for message in messages:
            if isinstance(message, str):
                socket.send_text(message)
            else:
                socket.send_json(message)

        terminal_count = 0
        start_time = time.monotonic()
        while terminal_count < expected_terminal:
            elapsed = time.monotonic() - start_time
            if elapsed > timeout_per_frame * expected_terminal + 2.0:
                missing = expected_terminal - terminal_count
                raise AssertionError(
                    f"Timeout after {elapsed:.1f}s: still waiting for {missing} terminal frame(s). "
                    f"Received {len(frames)} frames: {[f.get('type') for f in frames]}"
                )
            raw = _receive_or_none(socket)
            if raw is None:
                missing = expected_terminal - terminal_count
                raise AssertionError(
                    f"Socket closed prematurely: still waiting for {missing} terminal frame(s). "
                    f"Received {len(frames)} frames: {[f.get('type') for f in frames]}"
                )
            frame = json.loads(raw["text"])
            frames.append(frame)
            if frame.get("type") in TERMINAL_FRAMES:
                terminal_count += 1
    return frames


# --------------------------------------------------------------------------- #
# plain HTTP
# --------------------------------------------------------------------------- #


def test_health_reports_ok(client: Any) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_serves_html_mentioning_the_websocket_path(client: Any) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "/ws" in response.text
    assert "<html" in response.text.lower()


def test_index_advertises_the_wire_types_its_script_switches_on(client: Any) -> None:
    response = client.get("/")

    for expected in ("task_complete", "task_start", "run_end", "error"):
        assert expected in response.text


def test_api_start_accepts_a_run(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_run(monkeypatch)

    response = client.post("/api/start", json={"goal": "build a health endpoint"})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "accepted"
    assert body["goal"] == "build a health endpoint"
    assert body["stream_path"] == "/ws"
    assert body["run_id"].startswith("run-")


def test_api_start_rejects_an_empty_goal(client: Any) -> None:
    response = client.post("/api/start", json={"goal": ""})

    assert response.status_code == 422


def test_api_start_rejects_a_whitespace_goal(client: Any) -> None:
    response = client.post("/api/start", json={"goal": "   "})

    assert response.status_code == 422


async def test_api_start_defers_the_run_to_a_background_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The endpoint must hand the loop to Starlette rather than await it."""
    started: list[str] = []
    release = asyncio.Event()

    async def blocking_run(self: Any) -> RunResult:
        """Never finish until the test releases it.

        Args:
            self: The orchestrator instance.

        Returns:
            A green :class:`RunResult`.
        """
        started.append(self.goal)
        await release.wait()
        return RunResult(cycles=1, completed=True, stopped_reason="released")

    monkeypatch.setattr(web_module.Orchestrator, "run", blocking_run)
    background = BackgroundTasks()

    accepted = await web_module.api_start(StartRequest(goal="deferred goal"), background)

    assert accepted.status == "accepted"
    assert started == []
    assert len(background.tasks) == 1
    release.set()
    await background()
    assert started == ["deferred goal"]


def test_each_run_gets_its_own_settings_copy() -> None:
    first = web_module._settings_for(StartRequest(goal="a", model="gpt-4o-mini"))
    second = web_module._settings_for(StartRequest(goal="a"))

    assert first is not second
    assert first.override_model == "gpt-4o-mini"
    assert second.override_model != "gpt-4o-mini"


async def test_run_unobserved_swallows_orchestrator_errors() -> None:
    class _Exploding:
        """An orchestrator whose ``run`` always fails."""

        async def run(self) -> RunResult:
            """Fail loudly.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("background boom")

    await web_module._run_unobserved("run-1", _Exploding())


async def test_run_unobserved_reraises_cancellation() -> None:
    class _Cancelled:
        """An orchestrator whose ``run`` is cancelled."""

        async def run(self) -> RunResult:
            """Be cancelled.

            Raises:
                asyncio.CancelledError: Always.
            """
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await web_module._run_unobserved("run-1", _Cancelled())


# --------------------------------------------------------------------------- #
# websocket streaming
# --------------------------------------------------------------------------- #


def test_ws_emits_frames_with_snake_case_types(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug this locks down: the socket accepted, then never sent anything."""
    _install_run(
        monkeypatch,
        events=[
            _event(EventType.RUN_START, "Goal: build it"),
            _event(EventType.CYCLE_START, "Cycle 1"),
            _event(EventType.PLAN, "planned", task_count=1),
            _event(EventType.TASK_START, "doing it", task_id="1"),
            _event(EventType.TASK_COMPLETE, "done it", task_id="1", result="ok"),
            _event(EventType.TEST_RESULT, "suite green", passed=True),
            _event(EventType.CYCLE_END, "cycle finished", tests_passed=True),
        ],
        result=RunResult(
            cycles=1,
            completed=True,
            tasks_completed=1,
            stopped_reason="All 1 planned tasks completed and the test suite passed.",
        ),
    )

    frames = _collect_ws(client, [{"goal": "build it"}])

    types = [frame["type"] for frame in frames]
    assert types[0] == "run_start"
    assert types[-1] == "run_end"
    for expected in (
        "cycle_start",
        "plan",
        "task_start",
        "task_complete",
        "test_result",
        "cycle_end",
    ):
        assert expected in types, types
    task_start = next(frame for frame in frames if frame["type"] == "task_start")
    assert task_start["task_id"] == "1"
    assert task_start["message"] == "doing it"
    assert task_start["cycle"] == 1
    assert task_start["data"] == {}


def test_ws_final_frame_carries_the_run_result(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = RunResult(
        cycles=3,
        completed=False,
        tasks_completed=2,
        tasks_failed=1,
        cycles_failed=1,
        last_error="ValueError: bad plan",
        stopped_reason="Stopped after 3 consecutive repair cycles.",
    )
    _install_run(monkeypatch, result=result)

    frames = _collect_ws(client, [{"goal": "build it"}])

    final = frames[-1]
    assert final["type"] == "run_end"
    assert final["completed"] is False
    assert final["cycles"] == 3
    assert final["tasks_completed"] == 2
    assert final["tasks_failed"] == 1
    assert final["cycles_failed"] == 1
    assert final["last_error"] == "ValueError: bad plan"
    assert final["stopped_reason"] == result.stopped_reason
    assert final["run_id"].startswith("run-")


def test_ws_frames_share_one_run_id(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_run(monkeypatch, events=[_event(EventType.RUN_START, "hi")])

    frames = _collect_ws(client, [{"goal": "build it"}])

    assert len({frame["run_id"] for frame in frames}) == 1


def test_ws_forwards_the_requested_model(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str | None] = []

    async def fake_run(self: Any) -> RunResult:
        """Record the resolved override model.

        Args:
            self: The orchestrator instance.

        Returns:
            A green :class:`RunResult`.
        """
        seen.append(self.settings.override_model)
        return RunResult(cycles=1, completed=True, stopped_reason="ok")

    monkeypatch.setattr(web_module.Orchestrator, "run", fake_run)
    _collect_ws(client, [{"goal": "build it", "model": "gpt-4o-mini"}])

    assert seen == ["gpt-4o-mini"]


def test_ws_accepts_a_second_message_on_the_same_socket(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old endpoint read exactly one message and then closed."""
    goals = _install_run(monkeypatch, events=[_event(EventType.RUN_START, "hi")])

    frames = _collect_ws(client, [{"goal": "first goal"}, {"goal": "second goal"}])

    assert goals == ["first goal", "second goal"]
    assert [frame["type"] for frame in frames].count("run_end") == 2


def test_ws_second_run_reports_its_own_state(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"count": 0}

    async def fake_run(self: Any) -> RunResult:
        """Return a different verdict on each call.

        Args:
            self: The orchestrator instance.

        Returns:
            A green run the first time, a red one after.
        """
        calls["count"] += 1
        if calls["count"] == 1:
            return RunResult(cycles=1, completed=True, stopped_reason="first")
        return RunResult(cycles=2, completed=False, stopped_reason="second")

    monkeypatch.setattr(web_module.Orchestrator, "run", fake_run)

    frames = _collect_ws(client, [{"goal": "first goal"}, {"goal": "second goal"}])

    run_ends = [frame for frame in frames if frame["type"] == "run_end"]
    assert [frame["completed"] for frame in run_ends] == [True, False]
    assert [frame["stopped_reason"] for frame in run_ends] == ["first", "second"]
    assert run_ends[0]["run_id"] != run_ends[1]["run_id"]


def test_ws_malformed_json_produces_an_error_frame(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_run(monkeypatch)

    frames = _collect_ws(client, ["{not valid json"])

    assert len(frames) == 1
    assert frames[0]["type"] == "error"
    assert "malformed JSON" in frames[0]["message"]


def test_ws_blank_goal_produces_an_error_frame(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_run(monkeypatch)

    frames = _collect_ws(client, [{"goal": "   "}])

    assert frames[0]["type"] == "error"
    assert "empty" in frames[0]["message"]


def test_ws_missing_goal_key_produces_an_error_frame(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_run(monkeypatch)

    frames = _collect_ws(client, [{"nope": "nothing useful"}])

    assert frames[0]["type"] == "error"
    assert "goal" in frames[0]["message"]


def test_ws_a_raising_orchestrator_produces_an_error_frame(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_run(monkeypatch, error=RuntimeError("orchestrator exploded"))

    frames = _collect_ws(client, [{"goal": "build it"}])

    assert frames[0]["type"] == "error"
    assert "orchestrator exploded" in frames[0]["message"]


def test_ws_closes_after_a_client_error(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_run(monkeypatch)

    with client.websocket_connect("/ws") as socket:
        socket.send_text("{broken")
        assert json.loads(socket.receive()["text"])["type"] == "error"
        assert socket.receive()["type"] == "websocket.close"


def test_ws_disconnect_cancels_a_running_orchestrator(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    cancelled = threading.Event()

    async def hanging_run(self: Any) -> RunResult:
        """Hang until cancelled.

        Args:
            self: The orchestrator instance.

        Returns:
            Never returns normally.

        Raises:
            asyncio.CancelledError: When the endpoint cancels the run.
        """
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("the run should have been cancelled")

    monkeypatch.setattr(web_module.Orchestrator, "run", hanging_run)

    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": "build it"})
        assert started.wait(timeout=10)

    assert cancelled.wait(timeout=10)


# --------------------------------------------------------------------------- #
# regression tests for the frame-drop bug (fixed in furrow/web/server.py)
# --------------------------------------------------------------------------- #


def test_ws_accepts_goals_sent_back_to_back(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two goals sent before any receive; both must produce run_end in order."""
    goals = _install_run(
        monkeypatch, events=[_event(EventType.RUN_START, "hi")]
    )

    frames = _collect_ws(
        client, [{"goal": "first goal"}, {"goal": "second goal"}]
    )

    run_ends = [frame for frame in frames if frame["type"] == "run_end"]
    assert len(run_ends) == 2
    assert goals == ["first goal", "second goal"]
    assert run_ends[0]["run_id"] != run_ends[1]["run_id"]
    assert run_ends[0]["stopped_reason"] == "all done"
    assert run_ends[1]["stopped_reason"] == "all done"


def test_ws_accepts_a_goal_sent_while_a_run_is_streaming(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Goal B sent during A's ~0.3s run; A's run_end then B's run_end must arrive."""
    started_a = asyncio.Event()
    released_a = asyncio.Event()

    async def slow_run(self: Any) -> RunResult:
        """First run holds for ~0.3s; second run returns immediately."""
        if not started_a.is_set():
            started_a.set()
            await released_a.wait()
        return RunResult(cycles=1, completed=True, stopped_reason="done")

    monkeypatch.setattr(web_module.Orchestrator, "run", slow_run)

    async def run_test() -> list[dict[str, Any]]:
        with client.websocket_connect("/ws") as socket:
            socket.send_json({"goal": "goal A"})
            await asyncio.wait_for(started_a.wait(), timeout=5.0)
            socket.send_json({"goal": "goal B"})
            released_a.set()

            frames = []
            terminal_count = 0
            while terminal_count < 2:
                raw = socket.receive()
                if raw["type"] == "websocket.close":
                    raise AssertionError("Socket closed before both runs finished")
                frame = json.loads(raw["text"])
                frames.append(frame)
                if frame.get("type") in TERMINAL_FRAMES:
                    terminal_count += 1
            return frames

    frames = asyncio.run(run_test())

    run_ends = [frame for frame in frames if frame["type"] == "run_end"]
    assert len(run_ends) == 2
    assert run_ends[0]["run_id"] != run_ends[1]["run_id"]
    assert run_ends[0]["stopped_reason"] == "done"
    assert run_ends[1]["stopped_reason"] == "done"


def test_ws_handles_many_goals_in_sequence(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five goals back-to-back; all five run_end frames must arrive in order."""
    goals = _install_run(
        monkeypatch, events=[_event(EventType.RUN_START, "hi")]
    )

    frames = _collect_ws(
        client,
        [
            {"goal": "goal 1"},
            {"goal": "goal 2"},
            {"goal": "goal 3"},
            {"goal": "goal 4"},
            {"goal": "goal 5"},
        ],
    )

    run_ends = [frame for frame in frames if frame["type"] == "run_end"]
    assert len(run_ends) == 5
    assert goals == ["goal 1", "goal 2", "goal 3", "goal 4", "goal 5"]
    run_ids = [frame["run_id"] for frame in run_ends]
    assert len(set(run_ids)) == 5


def test_ws_does_not_lose_a_frame_arriving_during_the_gap_between_runs(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Send goal A, wait for A's run_end, then send goal B; B's run_end must arrive.

    This targets the specific idle window between runs where, on the old code,
    no reader task existed and a frame could be dropped.
    """
    goals = _install_run(
        monkeypatch, events=[_event(EventType.RUN_START, "hi")]
    )

    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": "goal A"})
        frames = []
        while True:
            raw = _receive_or_none(socket)
            if raw is None:
                raise AssertionError("Socket closed during first run")
            frame = json.loads(raw["text"])
            frames.append(frame)
            if frame.get("type") in TERMINAL_FRAMES:
                break

        socket.send_json({"goal": "goal B"})
        while True:
            raw = _receive_or_none(socket)
            if raw is None:
                raise AssertionError("Socket closed during second run")
            frame = json.loads(raw["text"])
            frames.append(frame)
            if frame.get("type") in TERMINAL_FRAMES:
                break

    run_ends = [frame for frame in frames if frame["type"] == "run_end"]
    assert len(run_ends) == 2
    assert goals == ["goal A", "goal B"]
    assert run_ends[0]["run_id"] != run_ends[1]["run_id"]


# --------------------------------------------------------------------------- #
# frame helpers
# --------------------------------------------------------------------------- #


def test_event_frame_preserves_every_event_field() -> None:
    event = FurrowEvent(
        type=EventType.TASK_COMPLETE,
        message="done",
        cycle=2,
        task_id="7",
        data={"result": "ok", "files_written": ["a.py"]},
    )

    frame = EventFrame.from_event(event, "run-abc")
    dumped = frame.model_dump(mode="json")

    assert dumped["type"] == "task_complete"
    assert dumped["message"] == "done"
    assert dumped["cycle"] == 2
    assert dumped["task_id"] == "7"
    assert dumped["data"] == {"result": "ok", "files_written": ["a.py"]}
    assert dumped["run_id"] == "run-abc"


def test_run_end_frame_uses_snake_case_wire_values() -> None:
    result = RunResult(cycles=2, completed=True, tasks_completed=1, stopped_reason="done")

    frame = RunEndFrame.from_result(result, "run-abc").model_dump(mode="json")

    assert frame["type"] == "run_end"
    assert frame["tasks_completed"] == 1
    assert frame["stopped_reason"] == "done"
    assert frame["run_id"] == "run-abc"


def test_error_frame_defaults() -> None:
    frame = ErrorFrame(message="broken").model_dump(mode="json")

    assert frame == {"type": "error", "message": "broken", "run_id": ""}


def test_start_request_requires_a_goal() -> None:
    with pytest.raises(ValueError):
        StartRequest(goal="")


def test_start_request_optional_model() -> None:
    assert StartRequest(goal="g").model is None


def test_public_host_binding_prints_a_warning(capsys: Any) -> None:
    web_module._warn_if_public("0.0.0.0")

    captured = capsys.readouterr()

    assert "NO authentication" in captured.err


def test_loopback_binding_prints_nothing(capsys: Any) -> None:
    web_module._warn_if_public("127.0.0.1")

    assert capsys.readouterr().err == ""
