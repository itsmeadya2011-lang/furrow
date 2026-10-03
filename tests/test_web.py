"""The web monitor: the health probe, the page, and the streaming websocket.

Nothing here talks to a provider or a shell: the ``Orchestrator`` and the
``LLMClient`` factory are both stubbed, so every assertion is about the
transport (what reaches the socket) and never about the loop's real work. Socket
reads are bounded, so a handler that forgets to close or to send can fail the
test instead of hanging it.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Iterator
from typing import Any, TypeVar

import pytest

# Aliased on purpose: a module-scope name starting with ``Test`` is collected by
# pytest as a test class.
from starlette.testclient import TestClient as Client
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from furrow import config
from furrow.web import server as server_module
from furrow.web.server import app, strip_markup

MODEL_FIELDS = ("model", "planner_model", "worker_model", "tester_model")

MODEL = "my-model"

T = TypeVar("T")


# ------------------------------------------------------------------ utilities


def _bounded(work: Callable[[], T], timeout: float, description: str) -> T:
    """Run ``work`` on a daemon thread and fail the test if it overruns.

    A websocket read blocks until the server speaks, so the wait has to be capped
    somewhere; doing it in a thread keeps the test synchronous while making a
    silent server an ordinary failure.
    """
    box: list[Any] = []

    def target() -> None:
        try:
            box.append(work())
        except BaseException as exc:  # re-raised on the calling thread
            box.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        pytest.fail(f"{description} did not finish within {timeout}s")
    (outcome,) = box
    if isinstance(outcome, BaseException):
        raise outcome
    return outcome


def drain_socket(
    socket: WebSocketTestSession, *, limit: int = 32, timeout: float = 5.0
) -> list[str]:
    """Every message received, up to the point where the server closes."""

    def collect() -> list[str]:
        received: list[str] = []
        while len(received) < limit:
            try:
                received.append(socket.receive_text())
            except WebSocketDisconnect:
                return received
        return received

    return _bounded(collect, timeout, "the websocket")


def next_message(socket: WebSocketTestSession, *, timeout: float = 5.0) -> str:
    """The next message, without waiting for the socket to close."""
    return _bounded(socket.receive_text, timeout, "a websocket message")


def emitter(messages: tuple[str, ...]) -> Callable[[Any], Awaitable[None]]:
    """A loop ``run`` that pushes ``messages`` through its ``on_event`` hook."""

    async def run(loop: Any) -> None:
        for message in messages:
            loop.on_event(message)

    return run


def install_loop(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run: Callable[[Any], Awaitable[None]] | None = None,
) -> list[Any]:
    """Replace ``Orchestrator`` with a stub; returns the stubs that get built.

    The stub refuses to be built without ``on_event``: the whole point of ``/ws``
    is that the loop's output reaches the browser, so a handler that runs the
    loop to completion while forwarding nothing must fail here.
    """
    made: list[Any] = []

    class StubLoop:
        def __init__(self, goal: str, client: Any = None, on_event: Any = None) -> None:
            assert on_event is not None, "/ws must stream loop output through on_event"
            self.goal = goal
            self.client = client
            self.on_event = on_event
            self.settings = getattr(client, "settings", None)
            made.append(self)

        async def run(self) -> None:
            if run is not None:
                await run(self)

    monkeypatch.setattr(server_module, "Orchestrator", StubLoop)
    return made


class RecordingClient:
    """Offline stand-in for the LLM client: records the settings it was handed."""

    def __init__(self, settings: Any = None) -> None:
        self.settings = settings if settings is not None else config.settings


@pytest.fixture(autouse=True)
def offline_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test in this file may build a real provider client."""
    monkeypatch.setattr(server_module, "LLMClient", RecordingClient)


@pytest.fixture
def client() -> Iterator[Client]:
    with Client(app) as test_client:
        yield test_client


# --------------------------------------------------------------------- http


def test_health_reports_ok(client: Client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_serves_the_monitor_page(client: Client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    # A page served over TLS has to open a secure socket, not a plaintext one.
    assert "location.protocol" in body
    assert "wss://" in body
    assert "ws://" in body
    assert "<style>" in body
    # The marker the page prints once the run is over.
    assert "--- Done ---" in body


def test_index_escapes_the_goal_query(client: Client) -> None:
    payload = "<script>alert(1)</script>"
    response = client.get("/", params={"goal": payload})
    assert response.status_code == 200
    assert payload not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text


def test_index_without_a_goal_leaves_the_field_empty(client: Client) -> None:
    response = client.get("/")
    assert 'id="goal"' in response.text
    assert 'value=""' in response.text


# -------------------------------------------------------------- strip_markup


@pytest.mark.parametrize(
    ("raw", "plain"),
    [
        ("[bold green]Furrow[/bold green]", "Furrow"),
        ("[green]ok[/]", "ok"),
        ("[bold]done[/bold]", "done"),
        ("[red]failed[/red]", "failed"),
        ("[bold red on white]loud[/bold red on white]", "loud"),
        ("[/]", ""),
        ("[bright_cyan]hi[/bright_cyan]", "hi"),
    ],
)
def test_strip_markup_removes_rich_tags(raw: str, plain: str) -> None:
    assert strip_markup(raw) == plain


@pytest.mark.parametrize(
    "text",
    [
        "a[0]",
        "array[i]",
        "Goal: [1, 2] and a[0]",
        "items[3]",
        "no markup at all",
    ],
)
def test_strip_markup_leaves_ordinary_brackets_alone(text: str) -> None:
    # A regex that ate any bracket would mangle these, which is exactly how task
    # descriptions and test names would get silently corrupted.
    assert strip_markup(text) == text


def test_strip_markup_removes_ansi_escapes() -> None:
    assert strip_markup("\x1b[32mgreen\x1b[0m") == "green"
    assert strip_markup("\x1b[1;33mwarn\x1b[0m done") == "warn done"


def test_strip_markup_accepts_non_strings() -> None:
    # The handler is defensive about whatever the loop hands it.
    value: Any = 42
    assert strip_markup(value) == "42"


# ------------------------------------------------------------------ websocket


def test_ws_streams_loop_output_to_the_client(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    # THE regression: /ws used to await the loop and never call ``on_event``, so
    # the browser received nothing at all until the run was over.
    stubs = install_loop(
        monkeypatch,
        run=emitter(("hello from the loop", "[bold green]done[/bold green]", "all set")),
    )
    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": "g"})
        messages = drain_socket(socket)

    assert [stub.goal for stub in stubs] == ["g"]
    assert messages[0] == "hello from the loop"
    assert "done" in messages
    assert "all set" in messages
    # Markup is stripped on the way to the browser, not rendered as text.
    assert [message for message in messages if "[" in message] == []


def test_ws_closes_the_socket_when_the_run_finishes(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = install_loop(monkeypatch, run=emitter(("only message",)))
    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": "g"})
        # ``drain_socket`` only returns once the server closed, so returning at
        # all is the assertion that the socket is not left dangling.
        messages = drain_socket(socket)

    assert messages == ["only message"]
    assert len(stubs) == 1


def test_ws_model_override_reaches_every_agent_without_mutating_the_singleton(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = install_loop(monkeypatch)
    before = {field: getattr(config.settings, field) for field in MODEL_FIELDS}
    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": "g", "model": MODEL})
        drain_socket(socket)

    assert len(stubs) == 1
    seen = stubs[0].settings
    assert seen is not config.settings
    for field in MODEL_FIELDS:
        assert getattr(seen, field) == MODEL, f"the override never reached {field}"
        assert getattr(config.settings, field) == before[field], (
            f"{field} leaked into the global singleton"
        )


def test_ws_without_a_model_passes_the_shared_settings(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = install_loop(monkeypatch)
    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": "g"})
        drain_socket(socket)

    assert len(stubs) == 1
    assert stubs[0].settings is server_module.global_settings


def test_ws_rejects_an_empty_goal(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    stubs = install_loop(monkeypatch)
    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": "   "})
        messages = drain_socket(socket)

    assert stubs == [], "an empty goal must not start a run"
    assert any("goal" in message for message in messages)
    assert messages[0].startswith("error:")


def test_ws_rejects_a_malformed_payload(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    stubs = install_loop(monkeypatch)
    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": 123})
        messages = drain_socket(socket)

    assert stubs == []
    assert messages[0].startswith("error:")


def test_ws_rejects_a_non_object_payload(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    stubs = install_loop(monkeypatch)
    with client.websocket_connect("/ws") as socket:
        socket.send_json([1, 2, 3])
        messages = drain_socket(socket)

    assert stubs == []
    assert messages[0].startswith("error:")


def test_ws_rejects_a_non_json_body(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    stubs = install_loop(monkeypatch)
    with client.websocket_connect("/ws") as socket:
        socket.send_text("not json")
        messages = drain_socket(socket)

    assert stubs == []
    assert messages[0].startswith("error:")


def test_ws_survives_a_payload_the_server_rejects(client: Client) -> None:
    """A rejected payload must not take the application down with it."""
    with client.websocket_connect("/ws") as socket:
        socket.send_text("not json")
        drain_socket(socket)

    assert client.get("/health").json() == {"status": "ok"}


def test_ws_survives_an_on_event_path_that_raises(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Unprintable:
        def __str__(self) -> str:
            raise ValueError("cannot render")

    async def run(loop: Any) -> None:
        loop.on_event("before the explosion")
        loop.on_event(Unprintable())
        loop.on_event("after the explosion")

    install_loop(monkeypatch, run=run)
    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": "g"})
        messages = drain_socket(socket)

    # The bad message is dropped; the run finishes and the socket closes.
    assert "before the explosion" in messages
    assert "after the explosion" in messages


def test_ws_reports_a_failing_run_and_closes(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run(loop: Any) -> None:
        loop.on_event("starting")
        raise RuntimeError("boom")

    install_loop(monkeypatch, run=run)
    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": "g"})
        messages = drain_socket(socket)

    assert "starting" in messages
    assert any("boom" in message for message in messages)
    assert client.get("/health").json() == {"status": "ok"}


def test_ws_disconnect_cancels_the_run(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    cancelled = threading.Event()
    finished_normally = threading.Event()

    async def run(loop: Any) -> None:
        started.set()
        try:
            while True:
                loop.on_event("still working")
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        else:  # pragma: no cover - only reached if the task was never cancelled
            finished_normally.set()

    install_loop(monkeypatch, run=run)
    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": "g"})
        assert started.wait(5), "the stubbed loop never started"
        assert next_message(socket) == "still working"
        socket.close()

    assert cancelled.wait(5), "the run task kept going after the client left"
    assert not finished_normally.is_set()


def test_ws_rejects_a_disconnect_before_the_start_payload(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = install_loop(monkeypatch)
    with client.websocket_connect("/ws") as socket:
        socket.close()

    assert stubs == []
