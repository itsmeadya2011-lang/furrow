"""Regression tests for :mod:`furrow.web.server`.

The websocket endpoint is driven through starlette's ``TestClient``, so no real socket is
opened. ``Orchestrator`` is replaced with a fake that emits a fixed script of event lines, so
the streaming contract can be asserted exactly (ordering, the terminal signal, validation
rejections, and cancellation when the client disappears mid-run).
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from furrow.config import settings as singleton_settings
from furrow.llm import LLMClient

server_module = importlib.import_module("furrow.web.server")

#: Rich tags must never reach a browser; the endpoint promises plain text.
RICH_TAGS = ("[bold", "[red", "[green", "[yellow", "[cyan")

#: The four role fields a ``model`` override replaces, per the endpoint's own list.
ROLE_FIELDS = ("model", "planner_model", "worker_model", "tester_model")

#: What the global settings singleton holds when a test pins every role field to a distinct
#: known value, so "the configured default" is never whatever the host environment happened to
#: set at import time.
CONFIGURED_DEFAULTS = tuple(f"configured-{field}" for field in ROLE_FIELDS)


class FakeOrchestrator:
    """Emits a fixed list of event lines, then returns."""

    created: list[FakeOrchestrator] = []
    lines: tuple[str, ...] = ("Goal: GOAL", "Cycle 1", "Task a completed")

    def __init__(self, goal: str, client: object = None, on_event: Any = None) -> None:
        self.goal = goal
        self.client = client
        self.on_event = on_event
        self.cancelled = False
        self.finished = False
        self.task: asyncio.Task[Any] | None = None
        FakeOrchestrator.created.append(self)

    async def run(self) -> None:
        self.task = asyncio.current_task()
        try:
            for line in type(self).lines:
                assert self.on_event is not None
                self.on_event(line)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.finished = True


class HangingOrchestrator(FakeOrchestrator):
    """Never finishes on its own: only a disconnect or cancellation ends the run."""

    async def run(self) -> None:
        self.task = asyncio.current_task()
        try:
            while True:
                assert self.on_event is not None
                self.on_event("tick")
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.finished = True


@pytest.fixture(autouse=True)
def fake_orchestrator(monkeypatch: pytest.MonkeyPatch) -> type[FakeOrchestrator]:
    FakeOrchestrator.created = []
    FakeOrchestrator.lines = ("Goal: GOAL", "Cycle 1", "Task a completed")
    monkeypatch.setattr(server_module, "Orchestrator", FakeOrchestrator)
    return FakeOrchestrator


@pytest.fixture
def client() -> TestClient:
    with TestClient(server_module.app) as test_client:
        yield test_client  # type: ignore[misc]


def drain(websocket: Any) -> list[str]:
    """Every line the server sends, up to the terminal close."""
    received: list[str] = []
    while True:
        try:
            received.append(websocket.receive_text())
        except WebSocketDisconnect:
            return received


def configure_distinct_role_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin each role-model field to its own value, so per-field mix-ups cannot hide."""
    for field in ROLE_FIELDS:
        monkeypatch.setattr(singleton_settings, field, f"configured-{field}")


def singleton_models() -> tuple[str, ...]:
    """The role models currently held by the process-global settings object."""
    return tuple(getattr(singleton_settings, field) for field in ROLE_FIELDS)


def run_models(index: int = 0) -> tuple[str, ...]:
    """The role models on the client the ``index``-th run was actually handed.

    Reading the client's own settings (rather than the global object) is the whole point: the
    run must see the per-request copy the endpoint builds, not the singleton it no longer
    touches.
    """
    client = FakeOrchestrator.created[index].client
    assert isinstance(client, LLMClient), "each run must get its own real LLMClient"
    return tuple(getattr(client.settings, field) for field in ROLE_FIELDS)


# ------------------------------------------------------------------------------- pages


def test_the_index_page_is_served(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Furrow" in response.text


# --------------------------------------------------------------------------- streaming


def test_events_stream_to_the_client_in_order(client: TestClient) -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "do the thing"})
        received = drain(websocket)

    assert received == list(FakeOrchestrator.lines)
    assert [item.goal for item in FakeOrchestrator.created] == ["do the thing"]


def test_the_socket_is_closed_after_the_run_ends(client: TestClient) -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "GOAL"})
        assert websocket.receive_text() == "Goal: GOAL"
        # The terminal signal: once the backlog is drained the server closes the socket, so
        # reading eventually raises instead of blocking forever.
        try:
            while True:
                websocket.receive_text()
        except WebSocketDisconnect:
            closed = True
        else:  # pragma: no cover - only reached if the server never closes
            closed = False

    assert closed is True


def test_events_are_plain_text_with_no_markup(client: TestClient) -> None:
    FakeOrchestrator.lines = ("Goal: [/]x", "Task [/]a completed")

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "GOAL"})
        received = drain(websocket)

    assert received == list(FakeOrchestrator.lines)
    for line in received:
        for tag in RICH_TAGS:
            assert tag not in line


def test_each_line_is_sent_as_its_own_message(client: TestClient) -> None:
    FakeOrchestrator.lines = tuple(f"line {index}" for index in range(5))

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "GOAL"})
        received = drain(websocket)

    assert received == [f"line {index}" for index in range(5)]


def test_a_failing_run_is_reported_to_the_client(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    class Exploding(FakeOrchestrator):
        async def run(self) -> None:
            raise RuntimeError("run blew up")

    monkeypatch.setattr(server_module, "Orchestrator", Exploding)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "GOAL"})
        received = drain(websocket)

    assert received == ["Error: run failed: run blew up"]


# -------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"goal": ""}, "Error: goal must not be empty."),
        ({"goal": "   "}, "Error: goal must not be empty."),
        ({}, "Error: invalid start request (goal: Field required)"),
        ({"goal": 5}, "Error: invalid start request (goal: Input should be a valid string)"),
        ([1, 2, 3], "Error: expected a JSON object with a 'goal' field."),
        ("plain text", "Error: expected a JSON object with a 'goal' field."),
        (None, "Error: expected a JSON object with a 'goal' field."),
    ],
)
def test_a_bad_payload_is_rejected_without_starting_a_run(
    client: TestClient, payload: object, expected: str
) -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json(payload)
        assert websocket.receive_text() == expected

    assert FakeOrchestrator.created == []


def test_the_goal_is_trimmed_before_it_is_used(client: TestClient) -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "  padded goal  "})
        drain(websocket)

    assert [item.goal for item in FakeOrchestrator.created] == ["padded goal"]


# --------------------------------------------------------------------- model override


def test_an_explicit_model_sets_all_four_role_models(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    # The override has to reach the run's own client. It used to "work" only because the
    # endpoint wrote it onto the process-global settings singleton and mutated every other
    # request along with it; the run is now built from a per-request copy.
    assert set(server_module.MODEL_FIELDS) == set(ROLE_FIELDS)
    configure_distinct_role_models(monkeypatch)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "GOAL", "model": "chosen-in-the-ui"})
        drain(websocket)

    assert run_models() == ("chosen-in-the-ui",) * len(ROLE_FIELDS)
    # ... and the global the run never touches is left exactly as configured.
    assert singleton_models() == CONFIGURED_DEFAULTS


def test_omitting_the_model_leaves_the_configured_models_alone(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    configure_distinct_role_models(monkeypatch)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "GOAL"})
        drain(websocket)

    assert run_models() == CONFIGURED_DEFAULTS
    assert singleton_models() == CONFIGURED_DEFAULTS


def test_a_model_chosen_by_one_client_does_not_leak_into_the_next_connection(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    # The leak this pins: the override was assigned onto the module-level settings singleton, so
    # a second browser tab that sent no model at all silently inherited the first one's override.
    # Three sequential connections through one server is the smallest reproduction.
    configure_distinct_role_models(monkeypatch)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "FIRST", "model": "MODEL-A"})
        drain(websocket)
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "SECOND"})
        drain(websocket)
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "THIRD", "model": "MODEL-B"})
        drain(websocket)

    assert run_models(0) == ("MODEL-A",) * len(ROLE_FIELDS)
    # The connection that asked for nothing must get the configured defaults, never MODEL-A.
    assert run_models(1) == CONFIGURED_DEFAULTS
    assert run_models(2) == ("MODEL-B",) * len(ROLE_FIELDS)
    assert singleton_models() == CONFIGURED_DEFAULTS
    # Distinct client objects, so no two runs could share mutable settings even by accident.
    assert len({id(run.client) for run in FakeOrchestrator.created}) == 3


# ------------------------------------------------------------------------ cancellation


def test_a_disconnect_cancels_the_orchestrator(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    monkeypatch.setattr(server_module, "Orchestrator", HangingOrchestrator)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "long run"})
        assert websocket.receive_text() == "tick"
    # Leaving the block closes the socket, which is the signal to stop spending tokens.

    assert len(HangingOrchestrator.created) == 1
    run = HangingOrchestrator.created[0]
    assert run.cancelled is True
    assert run.finished is True
    assert run.task is not None
    assert run.task.done(), "the orchestrator task must not outlive the socket"


def test_the_run_actually_stopped_early(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    # A hanging run would never end on its own, so returning from this test at all proves the
    # disconnect tore it down rather than the client waiting it out.
    monkeypatch.setattr(server_module, "Orchestrator", HangingOrchestrator)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "long run"})
        websocket.receive_text()

    assert HangingOrchestrator.created[0].cancelled is True


def test_no_orchestrator_task_is_left_running_after_a_normal_run(client: TestClient) -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "GOAL"})
        drain(websocket)

    run = FakeOrchestrator.created[0]
    assert run.cancelled is False
    assert run.finished is True
    assert run.task is not None and run.task.done()


def test_no_orchestrator_task_is_left_running_after_a_rejection(client: TestClient) -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": ""})
        websocket.receive_text()

    assert FakeOrchestrator.created == []


# -------------------------------------------------------------------------- defaults


def test_the_event_queue_and_drain_timeout_are_bounded() -> None:
    assert server_module.EVENT_QUEUE_SIZE == 500
    assert server_module.SOCKET_DRAIN_TIMEOUT == 5.0
