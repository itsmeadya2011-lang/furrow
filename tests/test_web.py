"""Tests for the Furrow web server.

Nothing here may touch the network and no real run may start: the orchestrator
is always replaced by :class:`FakeOrchestrator` and the settings point at a
temporary directory.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from furrow.config import RunResult, settings
from furrow.core.events import Event, EventType
from furrow.core.state import RunState, StateStore
from furrow.web import server


class FakeOrchestrator:
    """Stand-in for the real orchestrator: records kwargs, emits two events."""

    init_kwargs: ClassVar[dict[str, Any]] = {}
    instances: ClassVar[list[FakeOrchestrator]] = []

    def __init__(
        self,
        goal: str,
        client: Any = None,
        settings: Any = None,
        state_store: Any = None,
        event_bus: Any = None,
        resume: bool = False,
    ) -> None:
        FakeOrchestrator.instances.append(self)
        FakeOrchestrator.init_kwargs = {
            "goal": goal,
            "client": client,
            "settings": settings,
            "state_store": state_store,
            "event_bus": event_bus,
            "resume": resume,
        }
        self.goal = goal
        self.event_bus = event_bus
        self._status: dict[str, Any] = {
            "goal": goal,
            "original_goal": goal,
            "cycles": 0,
            "stopped_reason": "",
            "tasks": [],
        }

    def status(self) -> dict[str, Any]:
        return self._status

    async def run(self) -> RunResult:
        if self.event_bus is not None:
            self.event_bus.emit(Event(type=EventType.CYCLE_STARTED, data={"cycle": 1}))
            self.event_bus.emit(Event(type=EventType.TASK_FINISHED, data={"task": "1"}))
        return RunResult(
            passed=True,
            cycles=1,
            tasks_completed=1,
            summary="done",
            stopped_reason="goal complete",
        )


class HangingOrchestrator(FakeOrchestrator):
    """Fake orchestrator whose run never finishes on its own."""

    async def run(self) -> RunResult:
        await asyncio.sleep(30)
        raise AssertionError("the hanging run should have been cancelled")


class RecordingClient:
    """Fake LLM client that records whether it was closed."""

    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class ClientRecordingOrchestrator(FakeOrchestrator):
    """Fake orchestrator that owns a client recording its own closing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = RecordingClient()


class FailingOrchestrator(ClientRecordingOrchestrator):
    """Fake orchestrator whose run always fails."""

    async def run(self) -> RunResult:
        raise RuntimeError("boom")


class HangingClientOrchestrator(ClientRecordingOrchestrator):
    """Fake orchestrator whose run never finishes on its own."""

    async def run(self) -> RunResult:
        await asyncio.sleep(30)
        raise AssertionError("the hanging run should have been cancelled")


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Poll ``predicate`` until it holds or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


@pytest.fixture(autouse=True)
def isolated_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[Path]:
    """Point the server at a temporary workspace and reset the run registry."""
    run_settings = settings.model_copy(update={"workspace": tmp_path})
    monkeypatch.setattr(server, "settings", run_settings)
    server._reset_registry()
    yield tmp_path
    server._reset_registry()


@pytest.fixture
def fake_orchestrator(monkeypatch: pytest.MonkeyPatch) -> type[FakeOrchestrator]:
    """Replace the orchestrator in the web layer with the fake one."""
    FakeOrchestrator.init_kwargs = {}
    FakeOrchestrator.instances.clear()
    monkeypatch.setattr("furrow.web.server.Orchestrator", FakeOrchestrator)
    return FakeOrchestrator


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A test client bound to the Furrow web app."""
    with TestClient(server.app) as test_client:
        yield test_client


def test_index_returns_html(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Furrow" in response.text


def test_index_contains_ui_hooks(client: TestClient) -> None:
    """The served page exposes stable hooks for the new UI."""
    html = client.get("/").text

    # Goal input, Start/Resume buttons and the log still exist.
    for hook in (
        'id="goal"',
        'id="start"',
        'id="resume"',
        'id="log"',
    ):
        assert hook in html, f"missing hook {hook}"

    # New status header hooks.
    for hook in (
        'id="goal-display"',
        'id="cycle"',
        'id="state"',
        'id="stopped-reason"',
        'id="usage"',
    ):
        assert hook in html, f"missing hook {hook}"

    # Task panel container.
    assert 'id="tasks"' in html


def test_index_uses_text_content_for_untrusted_text(client: TestClient) -> None:
    """Task text is assigned via textContent, never innerHTML interpolation."""
    html = client.get("/").text

    assert "textContent" in html
    assert "innerHTML" not in html


def test_index_script_declares_every_element_reference(client: TestClient) -> None:
    """Every ``*El`` identifier used in the inline script is declared there.

    Regression test for the bug where ``renderStatus()`` referenced
    ``goalDisplayEl``, ``cycleEl``, ``stateEl``, ``stoppedReasonEl`` and
    ``usageEl`` without declaring them, so every call threw a ReferenceError
    that ``loadStatus()`` swallowed in its catch.
    """
    html = client.get("/").text
    script = html.split("<script>", 1)[1].rsplit("</script>", 1)[0]

    used = set(re.findall(r"\b([A-Za-z_$][A-Za-z0-9_$]*El)\b", script))
    declared = set(re.findall(r"\bconst\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=", script))

    undeclared = used - declared
    assert not undeclared, f"used but never declared: {sorted(undeclared)}"


def test_index_status_header_elements_bound_in_script(client: TestClient) -> None:
    """The script binds to every status header element present in the page."""
    html = client.get("/").text
    script = html.split("<script>", 1)[1].rsplit("</script>", 1)[0]

    for element_id in ("goal-display", "cycle", "state", "stopped-reason", "usage"):
        assert f"getElementById('{element_id}')" in script, (
            f"script does not bind to #{element_id}"
        )


def test_index_script_resets_summary_when_a_run_starts(client: TestClient) -> None:
    """Starting a run resets the summary tracker so every run logs its summary.

    Regression test for the bug where ``lastSummary`` was never reset: two
    consecutive runs ending with the same summary (e.g. "stopped: goal
    complete") produced only one summary line in the log.
    """
    html = client.get("/").text
    script = html.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    lines = script.splitlines()

    send_indexes = [index for index, line in enumerate(lines) if "socket.send(" in line]
    assert send_indexes, "the script never sends the start/resume message"

    for index in send_indexes:
        window = "\n".join(lines[max(0, index - 5) : index + 1])
        assert "lastSummary = null" in window, (
            "the summary tracker is not reset next to the start/resume message"
        )


def test_api_state_without_state(client: TestClient) -> None:
    response = client.get("/api/state")

    assert response.status_code == 200
    body = response.json()
    assert body["exists"] is False
    assert body["state"] is None


def test_api_state_with_state(client: TestClient, tmp_path: Path) -> None:
    store = StateStore(tmp_path / ".furrow" / "state.json")
    store.save(RunState(goal="ship it", original_goal="ship it", cycles=1))

    response = client.get("/api/state")

    assert response.status_code == 200
    body = response.json()
    assert body["exists"] is True
    assert body["state"]["original_goal"] == "ship it"


def test_api_start_returns_started(
    client: TestClient, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    response = client.post("/api/start", json={"goal": "build it"})

    assert response.status_code == 200
    body = response.json()
    assert body["started"] is True
    assert body["goal"] == "build it"
    kwargs = fake_orchestrator.init_kwargs
    assert kwargs["goal"] == "build it"
    assert kwargs["resume"] is False
    assert kwargs["event_bus"] is not None


def test_api_start_rejects_a_second_run(
    client: TestClient, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    server._CURRENT["running"] = True

    response = client.post("/api/start", json={"goal": "build it"})

    assert response.status_code == 409
    body = response.json()
    assert body["started"] is False
    assert body["error"] == "a run is already in progress"


def test_api_cancel_cancels_a_running_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling a running run stops it and clears the running flag."""
    monkeypatch.setattr("furrow.web.server.Orchestrator", HangingOrchestrator)

    client.post("/api/start", json={"goal": "build it"})

    assert _wait_until(lambda: server._CURRENT.get("running") is True), "the run never started"

    response = client.post("/api/cancel")
    assert response.status_code == 200
    body = response.json()
    assert body["cancelled"] is True

    run_task = server._CURRENT.get("task")
    assert run_task is not None
    assert _wait_until(lambda: bool(run_task.cancelled())), "the run was not cancelled"
    assert server._CURRENT.get("running") is False


def test_api_cancel_when_no_run_in_progress(
    client: TestClient,
) -> None:
    """Cancelling when nothing is running reports cancelled=false."""
    response = client.post("/api/cancel")
    assert response.status_code == 200
    body = response.json()
    assert body["cancelled"] is False
    assert body["error"] == "no run in progress"


def test_api_status(client: TestClient, fake_orchestrator: type[FakeOrchestrator]) -> None:
    client.post("/api/start", json={"goal": "build it"})

    response = client.get("/api/status")

    assert response.status_code == 200
    body = response.json()
    assert "running" in body
    assert "status" in body
    assert "result" in body


def test_websocket_streams_events(
    client: TestClient, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    messages: list[dict[str, Any]] = []

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "g"})
        for _ in range(20):
            try:
                message = ws.receive_json()
            except WebSocketDisconnect:
                break
            messages.append(message)
            if message.get("type") == "done":
                break

    events = [message for message in messages if message.get("type") != "done"]
    assert events, f"expected at least one event message, got {messages}"
    assert any(message.get("type") == "cycle_started" for message in events)
    assert any(message.get("type") == "task_finished" for message in events)

    done = messages[-1]
    assert done["type"] == "done"
    assert done["message"] == "done"
    assert done["data"]["passed"] is True


def test_websocket_requires_goal(
    client: TestClient, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    messages: list[dict[str, Any]] = []

    with client.websocket_connect("/ws") as ws:
        ws.send_json({})
        try:
            messages.append(ws.receive_json())
        except WebSocketDisconnect:
            messages.append({"type": "closed"})

    assert messages, "the server must answer instead of hanging"
    assert "error" in json.dumps(messages[-1]).lower()


def test_websocket_rejects_a_second_run(
    client: TestClient, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    # The registry is reset again by the ``isolated_server`` fixture.
    server._CURRENT["running"] = True

    messages: list[dict[str, Any]] = []
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "x"})
        try:
            messages.append(ws.receive_json())
        except WebSocketDisconnect:
            messages.append({"type": "closed"})

    assert messages, "the server must answer instead of hanging"
    assert messages[0]["type"] == "error"
    assert "already in progress" in messages[0]["message"]
    assert fake_orchestrator.instances == [], "no second run may be started"
    assert server._CURRENT["running"] is True, "the running run must not be touched"


def test_websocket_rejects_a_negative_max_cycles(
    client: TestClient, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    """The websocket path cannot smuggle an out-of-range ``max_cycles`` past the request model.

    The payload is rejected with an error frame and no run is started.
    """
    messages: list[dict[str, Any]] = []

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "x", "max_cycles": -1})
        try:
            messages.append(ws.receive_json())
        except WebSocketDisconnect:
            messages.append({"type": "closed"})

    assert messages, "the server must answer instead of hanging"
    assert messages[0]["type"] == "error"
    assert "max_cycles" in messages[0]["message"]
    assert fake_orchestrator.instances == [], "no run may be started"
    assert server._CURRENT.get("running") is False


def test_websocket_applies_a_valid_max_cycles_override(
    client: TestClient, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    """A valid ``max_cycles`` override reaches the orchestrator of the run."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "g", "max_cycles": 2})
        for _ in range(20):
            try:
                message = ws.receive_json()
            except WebSocketDisconnect:
                break
            if message.get("type") == "done":
                break

    assert fake_orchestrator.instances, "the run must have started"
    run_settings = fake_orchestrator.init_kwargs["settings"]
    assert run_settings.max_cycles == 2


def test_api_status_reports_not_running_after_run(
    client: TestClient, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    client.post("/api/start", json={"goal": "build it"})

    assert _wait_until(lambda: client.get("/api/status").json()["running"] is False), (
        "the run never finished"
    )
    body = client.get("/api/status").json()
    assert body["running"] is False
    assert body["result"] is not None


def test_websocket_disconnect_clears_the_running_flag(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("furrow.web.server.Orchestrator", HangingOrchestrator)

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "x"})
        assert _wait_until(lambda: server._CURRENT.get("running") is True), "the run never started"

    # The client is gone: the run must be cancelled and the flag cleared.
    assert _wait_until(lambda: client.get("/api/status").json()["running"] is False), (
        "the running flag stayed stuck after the client disconnected"
    )
    body = client.get("/api/status").json()
    assert body["running"] is False
    run_task = server._CURRENT.get("task")
    assert run_task is not None
    assert _wait_until(lambda: bool(run_task.cancelled())), "the run was not cancelled"


def _inline_script(client: TestClient) -> str:
    """The inline script of the served page."""
    html = client.get("/").text
    return html.split("<script>", 1)[1].rsplit("</script>", 1)[0]


def test_index_script_never_concatenates_a_raw_status_into_a_class_name(
    client: TestClient,
) -> None:
    """The status class is built from a fixed map, never from the raw status.

    The task status is planned by the LLM, so it is untrusted input: it must
    never end up verbatim in a class name.
    """
    script = _inline_script(client)

    forbidden = (
        r"'status-'\s*\+\s*status\b",
        r'"status-"\s*\+\s*status\b',
        r"'[^']*status-'\s*\+\s*status\b",
        r'"[^"]*status-"\s*\+\s*status\b',
        r"status-\$\{",
        r"className\s*=\s*[^;]*?\+\s*status\s*;",
    )
    for pattern in forbidden:
        assert not re.search(pattern, script), (
            f"the script concatenates a raw status into a class name: {pattern}"
        )

    assert re.search(r"statusEl\.className\s*=\s*[^;]*statusClass\(", script), (
        "the status class is not built from statusClass()"
    )
    assert re.search(r"statusEl\.textContent\s*=", script), (
        "the status text is not written with textContent"
    )


def test_index_script_knows_every_status_and_falls_back_to_unknown(
    client: TestClient,
) -> None:
    """The safe status map lists the known statuses and falls back to unknown."""
    script = _inline_script(client)

    match = re.search(r"const\s+SAFE_STATUSES\s*=\s*\{(.*?)\};", script, re.DOTALL)
    assert match is not None, "the script has no SAFE_STATUSES map"
    body = match.group(1)

    for status in ("pending", "running", "completed", "failed", "blocked", "skipped"):
        assert re.search(rf"\b{status}\s*:", body), f"{status} is missing from the status map"

    assert re.search(r"\bunknown\s*:", body), "the status map has no unknown entry"
    assert "return 'unknown';" in script, "statusClass() has no unknown fallback"


@pytest.fixture
def recording_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> type[ClientRecordingOrchestrator]:
    """Replace the orchestrator with a fake that owns a closable client."""
    FakeOrchestrator.instances.clear()
    monkeypatch.setattr("furrow.web.server.Orchestrator", ClientRecordingOrchestrator)
    return ClientRecordingOrchestrator


def test_api_start_releases_the_llm_client_when_the_run_finishes(
    client: TestClient, recording_orchestrator: type[ClientRecordingOrchestrator]
) -> None:
    """A finished background run releases the LLM client of the orchestrator."""
    response = client.post("/api/start", json={"goal": "build it"})

    assert response.status_code == 200
    assert len(recording_orchestrator.instances) == 1
    recorder = cast(RecordingClient, recording_orchestrator.instances[0].client)
    assert _wait_until(lambda: recorder.closed), "the run never released the client"


async def test_run_orchestrator_releases_the_client_when_the_run_raises() -> None:
    """A failing run still releases the client: the release lives in a finally."""
    orchestrator = FailingOrchestrator(goal="build it")

    with pytest.raises(RuntimeError):
        await server._run_orchestrator(cast(Any, orchestrator))

    assert orchestrator.client.closed is True


async def test_run_orchestrator_releases_the_client_when_the_run_is_cancelled() -> None:
    """A cancelled run (e.g. the websocket client left) releases the client."""
    orchestrator = HangingClientOrchestrator(goal="build it")

    task = asyncio.create_task(server._run_orchestrator(cast(Any, orchestrator)))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert orchestrator.client.closed is True


def test_index_contains_status_skipped_css_rule(client: TestClient) -> None:
    html = client.get("/").text
    assert ".status-skipped" in html


def test_start_request_rejects_negative_max_cycles(client: TestClient) -> None:
    response = client.post("/api/start", json={"goal": "x", "max_cycles": -1})
    assert response.status_code == 422


def test_api_start_rejects_a_negative_max_cycles_without_starting_a_run(
    client: TestClient, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    """An out-of-range ``max_cycles`` never reaches the orchestrator."""
    response = client.post("/api/start", json={"goal": "x", "max_cycles": -1})

    assert response.status_code == 422
    assert fake_orchestrator.instances == [], "no run may be started"
    assert server._CURRENT.get("running") is False


def test_run_settings_applies_a_valid_max_cycles_override() -> None:
    """A valid override is applied to the settings of the run."""
    original = server.settings.max_cycles

    run_settings = server._run_settings(3)

    assert run_settings.max_cycles == 3
    assert server.settings.max_cycles == original, "the shared settings must not be modified"


def test_run_settings_without_override_returns_the_current_settings() -> None:
    """Without an override the current settings are used as they are."""
    assert server._run_settings(None) is server.settings


def test_run_settings_rejects_an_out_of_range_max_cycles() -> None:
    """An out-of-range override is rejected instead of silently accepted.

    ``model_copy(update=...)`` skips the pydantic validation, so a negative
    ``max_cycles`` used to build a ``Settings`` object that violates the
    bounds of ``furrow.config.Settings``.
    """
    with pytest.raises(ValidationError):
        server._run_settings(-1)
