"""Tests for :mod:`furrow.web.server`.

The headline regression covered here is the old ``/ws`` handler, which contained no
``websocket.send`` calls at all: the browser showed "Starting..." and then "Closed."
while the run printed to a console nobody was watching. The streaming tests below therefore
assert the *whole* forwarded event sequence, not just that a socket opened.

Two layers are covered:

* end to end, with :class:`fastapi.testclient.TestClient` against the module-level ``app``.
  The loop runs on the ``stub_agents`` doubles, so no LLM, subprocess or wall clock is
  involved and the event sequence is finite by construction;
* unit level, calling ``_send``/``_drive``/``_reap``/``_settings_for`` directly under the
  ``bounded`` deadline fixture, so a websocket whose peer vanished or an orchestrator that
  crashed is asserted on without a portal thread in the way.

Two notes on the test client. The websockets of this starlette version run the app on a
portal thread, so an ``asyncio.Event`` cannot be shared with the test; ``threading.Event`` is
used for every cross-thread hand-off. And ``WebSocketTestSession.receive`` blocks with no
timeout of its own, so ``bound_receives`` wraps the portal call: a server that stops streaming
fails the test after a few seconds instead of hanging the suite forever.

``web_environment`` and ``client`` are module-local rather than conftest fixtures: this session
does not own ``tests/conftest.py``, so promoting them is left to whoever does.

``pytest`` collects classes named ``Test*``; no production ``Test*`` name is imported here
(``conftest.py`` already opts ``TestResult`` out for the whole suite).
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import threading
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from furrow.config import Settings
from furrow.core.orchestrator import LoopResult, Orchestrator
from furrow.web import server
from tests.conftest import AgentStubs, FakeLLMClient, build_settings, failing, make_plan, passing

#: ``captured_console`` silences the orchestrator's rich console; ``web_environment`` keeps
#: every settings-derived path inside ``tmp_path`` instead of the repository.
pytestmark = pytest.mark.usefixtures("captured_console", "web_environment")

StubInstaller = Callable[..., AgentStubs]

#: Every websocket read gets a deadline, because ``receive`` blocks without one.
RECEIVE_DEADLINE = 5.0

#: How long a cross-thread cancellation hand-off may take before the test gives up.
CANCEL_DEADLINE = 5.0

#: The exact event sequence of one cycle with one task and passing tests.
ONE_PASSING_CYCLE = [
    "run.start",
    "cycle.start",
    "plan",
    "task.start",
    "task.done",
    "test.result",
    "cycle.done",
    "run.done",
]

#: The event sequence that precedes a worker that never returns.
UNTIL_TASK_START = ["run.start", "cycle.start", "plan", "task.start"]


# --------------------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------------------


@pytest.fixture
def web_environment(monkeypatch: pytest.MonkeyPatch, tmp_workspace: Path) -> None:
    """Point the settings the web layer builds at a temp workspace with no real credentials.

    ``_settings_for`` builds a fresh ``Settings`` from the environment, so the workspace has to
    be redirected here rather than through an argument: a run started over a websocket must never
    touch the repository it was started from.
    """
    monkeypatch.setenv("FURROW_WORKSPACE", str(tmp_workspace))
    monkeypatch.setenv("FURROW_ANTHROPIC_API_KEY", "web-test-key")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "1")


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A test client for the module-level app (one portal for every request)."""
    with TestClient(server.app) as test_client:
        yield test_client


@dataclass
class FakeWebSocket:
    """The part of ``starlette.websockets.WebSocket`` that ``_send`` actually uses.

    ``failure`` reproduces what starlette raises once the peer is gone: ``WebSocketDisconnect``
    on an abrupt close, ``WebSocketDisconnected`` (a ``RuntimeError``) after a local close and
    ``OSError`` when the transport dies.
    """

    sent: list[dict[str, Any]] = field(default_factory=list)
    failure: BaseException | None = None

    async def send_json(self, data: dict[str, Any]) -> None:
        if self.failure is not None:
            raise self.failure
        self.sent.append(data)


def bound_receives(
    monkeypatch: pytest.MonkeyPatch,
    session: WebSocketTestSession,
    timeout: float = RECEIVE_DEADLINE,
) -> None:
    """Give the session's blocking portal calls a deadline so a stalled stream cannot hang.

    ``WebSocketTestSession.receive`` blocks forever when the server sends nothing, and the
    session runs the app on a portal thread that the ``bounded`` fixture cannot reach.
    Wrapping ``portal.call`` in ``asyncio.wait_for`` turns "the stream stopped" into a
    ``TimeoutError`` raised in the test.
    """
    portal_call = session.portal.call

    async def within(func: Callable[..., Awaitable[Any]], *args: Any) -> Any:
        return await asyncio.wait_for(func(*args), timeout)

    def call(func: Callable[..., Awaitable[Any]], *args: Any) -> Any:
        return portal_call(within, func, *args)

    monkeypatch.setattr(session.portal, "call", call, raising=False)


def script_source(page: str) -> str:
    """The page's inline script, with its standalone comment lines removed.

    The page documents its own ``textContent``-only rule in a comment that names the very sinks
    the tests below forbid, so a comment must not count as a use of one.
    """
    script = page.split("<script>", 1)[1].split("</script>", 1)[0]
    return "\n".join(line for line in script.splitlines() if not line.strip().startswith("//"))


def drain_until(session: WebSocketTestSession, terminal: str) -> list[dict[str, Any]]:
    """Read JSON messages until ``terminal`` arrives.

    Every run driven by the ``stub_agents`` doubles terminates, so the read count is finite;
    the deadline added by ``bound_receives`` is what stops a regressed server from hanging.
    """
    messages: list[dict[str, Any]] = []
    while True:
        message = session.receive_json()
        messages.append(message)
        if message["type"] == terminal:
            return messages


def event_types(messages: Sequence[dict[str, Any]]) -> list[str]:
    return [str(message["type"]) for message in messages]


def only_message(messages: Sequence[dict[str, Any]], event: str) -> dict[str, Any]:
    matches = [message for message in messages if message["type"] == event]
    assert len(matches) == 1, f"expected exactly one {event}, got {event_types(messages)}"
    return matches[0]


def spy_orchestrators(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every goal the web layer builds an orchestrator for, without running one."""
    goals: list[str] = []

    def fake(**kwargs: Any) -> Any:
        goals.append(str(kwargs.get("goal")))
        raise AssertionError("no run was expected to start")

    monkeypatch.setattr(server, "Orchestrator", fake)
    return goals


def record_llm_clients(monkeypatch: pytest.MonkeyPatch) -> list[FakeLLMClient]:
    """Capture the settings each run was given without ever completing a request."""
    clients: list[FakeLLMClient] = []

    def build(settings: Settings | None = None) -> FakeLLMClient:
        client = FakeLLMClient(settings or build_settings())
        clients.append(client)
        return client

    monkeypatch.setattr(server, "LLMClient", build)
    return clients


@dataclass
class WatchedRun:
    """Cross-thread view of a run that parks a worker until it is cancelled."""

    parked: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    results: list[LoopResult] = field(default_factory=list)


def watch_a_parked_run(stub_agents: StubInstaller, monkeypatch: pytest.MonkeyPatch) -> WatchedRun:
    """Install a run whose only worker never finishes, and record its cancellation.

    The worker parks on an ``asyncio.Event`` that nobody sets, so the run stays alive until
    something cancels it, and the only thing that can do that is the websocket layer noticing
    the client left. Reaching the park is therefore proof that the stream really was running.
    """
    watched = WatchedRun()

    async def park(_task: Any) -> None:
        watched.parked.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            watched.cancelled.set()
            raise

    stub_agents(plans=[make_plan("an endless task")], test_results=[passing()], worker_hook=park)

    class RecordingOrchestrator(Orchestrator):
        async def run(self) -> LoopResult:
            result = await super().run()
            watched.results.append(result)
            return result

    monkeypatch.setattr(server, "Orchestrator", RecordingOrchestrator)
    return watched


# --------------------------------------------------------------------------------------
# The single page
# --------------------------------------------------------------------------------------


def test_index_serves_html(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_index_offers_a_goal_input_and_a_start_button(client: TestClient) -> None:
    body = client.get("/").text

    assert 'id="goal"' in body
    assert 'id="submit"' in body
    assert "<title>Furrow</title>" in body


def test_index_writes_model_and_filesystem_text_with_textcontent(client: TestClient) -> None:
    """Every model-, filesystem- or error-derived string is rendered with ``textContent``.

    The goal, the plan rationale, task descriptions and test failures all come from a model or
    from disk, so this page is an XSS vector if any of them is ever interpolated into markup.
    """
    script = script_source(client.get("/").text)

    assert "row.textContent = text" in script
    assert "status.textContent = text" in script


@pytest.mark.parametrize(
    "sink", ["innerHTML", "insertAdjacentHTML", "document.write", "outerHTML"]
)
def test_index_never_assigns_untrusted_text_as_markup(client: TestClient, sink: str) -> None:
    script = script_source(client.get("/").text)

    assert sink not in script, f"{sink} would execute text that came from the model or disk"


def test_index_is_a_static_page(client: TestClient) -> None:
    """No request may be interpolated into the HTML: the goal can only enter over the socket."""
    assert client.get("/").text == server._INDEX_HTML


# --------------------------------------------------------------------------------------
# Streaming a run over the websocket
# --------------------------------------------------------------------------------------


def test_ws_streams_events_until_run_done(
    stub_agents: StubInstaller,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression test: every orchestrator event reaches the browser, in order.

    Before the rewrite the handler never called ``websocket.send``, so the browser saw the
    start and then the close while the run printed to an unwatched console.
    """
    stub_agents(plans=[make_plan("write the feature")], test_results=[passing()])

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_json({"goal": "ship the feature"})
        messages = drain_until(session, "run.done")

    assert len(messages) > 1, "a run must produce a stream, not a single message"
    assert event_types(messages) == ONE_PASSING_CYCLE


def test_ws_every_message_carries_a_type_and_a_message(
    stub_agents: StubInstaller,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_agents(plans=[make_plan("x")], test_results=[passing()])

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_json({"goal": "ship it"})
        messages = drain_until(session, "run.done")

    for message in messages:
        assert isinstance(message["type"], str) and message["type"]
        assert isinstance(message["message"], str) and message["message"], message


def test_ws_every_message_carries_a_cycle_key(
    stub_agents: StubInstaller,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The browser tracks the current cycle from this key, so it must always be present."""
    stub_agents(plans=[make_plan("x")], test_results=[passing()])

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_json({"goal": "ship it"})
        messages = drain_until(session, "run.done")

    assert all("cycle" in message for message in messages)
    assert only_message(messages, "cycle.start")["cycle"] == 1
    assert only_message(messages, "run.start")["cycle"] is None


def test_ws_run_start_carries_the_goal(
    stub_agents: StubInstaller,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_agents(plans=[make_plan("x")], test_results=[passing()])

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_json({"goal": "add jwt auth"})
        messages = drain_until(session, "run.start")

    assert messages[0]["goal"] == "add jwt auth"
    assert "add jwt auth" in messages[0]["message"]


def test_ws_run_done_carries_the_stop_reason_and_cycle_count(
    stub_agents: StubInstaller,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_agents(plans=[make_plan("x")], test_results=[passing()])

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_json({"goal": "ship it"})
        done = only_message(drain_until(session, "run.done"), "run.done")

    assert done["reason"] == "goal-complete"
    assert done["cycles"] == 1


def test_ws_run_done_reports_an_unfinished_run_as_such(
    stub_agents: StubInstaller,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that hits the cycle limit must not be dressed up as a finished goal."""
    stub_agents(plans=[make_plan("x")], test_results=[failing("still red")])

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_json({"goal": "ship it"})
        done = only_message(drain_until(session, "run.done"), "run.done")

    assert done["reason"] == "max-cycles"


def test_ws_test_result_message_preserves_the_verdict_and_failures(
    stub_agents: StubInstaller,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The browser renders ``passed`` and ``failures``; a lossy payload hides the verdict."""
    failures = ["tests/test_auth.py::test_login", "assert 401 == 200"]
    stub_agents(plans=[make_plan("x")], test_results=[failing(*failures)])

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_json({"goal": "ship it"})
        messages = drain_until(session, "run.done")

    verdict = only_message(messages, "test.result")
    assert verdict["passed"] is False
    assert verdict["failures"] == failures
    assert verdict["summary"] == "boom"
    assert verdict["message"] == "Tests failed: boom"


def test_ws_a_passing_test_result_is_reported_as_passed(
    stub_agents: StubInstaller,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_agents(plans=[make_plan("x")], test_results=[passing("all green")])

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_json({"goal": "ship it"})
        messages = drain_until(session, "run.done")

    verdict = only_message(messages, "test.result")
    assert verdict["passed"] is True
    assert verdict["failures"] == []
    assert verdict["message"] == "Tests passed: all green"


def test_ws_task_done_message_carries_the_task_id(
    stub_agents: StubInstaller,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_agents(plans=[make_plan("x")], test_results=[passing()])

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_json({"goal": "ship it"})
        messages = drain_until(session, "run.done")

    done = only_message(messages, "task.done")
    assert done["task_id"] == "1"
    assert done["status"] == "completed"
    assert isinstance(done["files"], list)


def test_ws_a_completed_run_is_closed_by_the_server(
    stub_agents: StubInstaller,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The browser's ``onclose`` must fire, otherwise the button stays disabled forever."""
    stub_agents(plans=[make_plan("x")], test_results=[passing()])

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_json({"goal": "ship it"})
        drain_until(session, "run.done")
        with pytest.raises(WebSocketDisconnect) as closed:
            session.receive_json()

    assert closed.value.code == 1000


def test_ws_a_model_override_reaches_every_role(
    stub_agents: StubInstaller,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One model must apply to all three roles, exactly like ``furrow start --model``."""
    stub_agents(plans=[make_plan("x")], test_results=[passing()])
    clients = record_llm_clients(monkeypatch)

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_json({"goal": "ship it", "model": "  house-model  "})
        drain_until(session, "run.done")

    run_settings = clients[0].settings
    assert run_settings.model == "house-model", "the model must be stripped before use"
    assert run_settings.planner_model == "house-model"
    assert run_settings.worker_model == "house-model"
    assert run_settings.tester_model == "house-model"


def test_ws_a_run_without_a_model_keeps_the_configured_role_models(
    stub_agents: StubInstaller,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_agents(plans=[make_plan("x")], test_results=[passing()])
    clients = record_llm_clients(monkeypatch)

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_json({"goal": "ship it", "model": "   "})
        drain_until(session, "run.done")

    defaults = build_settings()
    assert clients[0].settings.planner_model == defaults.planner_model
    assert clients[0].settings.worker_model == defaults.worker_model
    assert clients[0].settings.tester_model == defaults.tester_model


def test_ws_a_failing_send_does_not_stop_the_stream(
    stub_agents: StubInstaller,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead socket is the browser's problem: the run must keep going and still report done.

    ``Orchestrator._emit`` swallows a raising handler, so one undeliverable event may only
    cost that event, never the rest of the run.
    """
    stub_agents(plans=[make_plan("x")], test_results=[passing()])
    real_send = server._send

    async def flaky(websocket: Any, name: str, payload: dict[str, Any] | None = None) -> bool:
        if name == "plan":
            raise RuntimeError("the socket died mid-stream")
        return await real_send(websocket, name, payload)

    monkeypatch.setattr(server, "_send", flaky)

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_json({"goal": "ship it"})
        messages = drain_until(session, "run.done")

    types = event_types(messages)
    assert "plan" not in types, "the broken event must be the only thing lost"
    assert types[0] == "run.start"
    assert types[-1] == "run.done"
    assert "task.done" in types


# --------------------------------------------------------------------------------------
# Rejected start requests never start a run
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"goal": ""},
        {"goal": "   "},
        {"goal": "\n\t "},
        {"goal": None},
        {},
    ],
    ids=["empty", "whitespace", "newlines", "null", "missing"],
)
def test_ws_a_blank_goal_is_rejected_without_starting_a_run(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
) -> None:
    goals = spy_orchestrators(monkeypatch)

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_json(payload)
        message = session.receive_json()

    assert message["type"] == "error", f"{payload} must be rejected"
    assert "goal is required" in message["message"]
    assert goals == [], f"{payload} must not start a run"
    assert not server._ACTIVE_RUNS, f"{payload} must not leave a run behind"


def test_ws_malformed_json_is_rejected_without_starting_a_run(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    goals = spy_orchestrators(monkeypatch)

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_text("this is not json")
        message = session.receive_json()

    assert message["type"] == "error"
    assert "Malformed start request" in message["message"]
    assert goals == []
    assert not server._ACTIVE_RUNS, "a rejected request must not leave a run behind"


def test_ws_a_json_array_is_rejected_without_starting_a_run(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    goals = spy_orchestrators(monkeypatch)

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_json(["a goal", "another"])
        message = session.receive_json()

    assert message["type"] == "error"
    assert "JSON object with a goal" in message["message"]
    assert goals == []
    assert not server._ACTIVE_RUNS, "a rejected request must not leave a run behind"


# --------------------------------------------------------------------------------------
# Disconnecting cancels the run: a leaked task burns API credits nobody is watching
# --------------------------------------------------------------------------------------


def test_ws_a_disconnect_cancels_the_run_it_started(
    stub_agents: StubInstaller,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watched = watch_a_parked_run(stub_agents, monkeypatch)

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_json({"goal": "an endless goal"})
        messages = drain_until(session, "task.start")

        assert watched.parked.is_set(), "the worker never started, so nothing was streaming"
        assert event_types(messages) == UNTIL_TASK_START

        session.close()
        assert watched.cancelled.wait(CANCEL_DEADLINE), "the run kept going after the client left"

        trailing = session.receive_json()
        assert trailing["type"] == "run.done"
        assert trailing["reason"] == "cancelled"

    assert [result.stopped_reason for result in watched.results] == ["cancelled"]
    assert watched.results[0].passed is False


def test_ws_a_disconnect_leaves_no_active_run_registered(
    stub_agents: StubInstaller,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_ACTIVE_RUNS`` holds a strong reference for the whole run; it must end up empty."""
    watched = watch_a_parked_run(stub_agents, monkeypatch)

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_json({"goal": "an endless goal"})
        drain_until(session, "task.start")

        assert len(server._ACTIVE_RUNS) == 1, "a live run must be reachable, not collectable"

        session.close()
        assert watched.cancelled.wait(CANCEL_DEADLINE)

    assert not server._ACTIVE_RUNS, "a cancelled run must be released"


def test_ws_a_finished_run_leaves_no_active_run_registered(
    stub_agents: StubInstaller,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_agents(plans=[make_plan("x")], test_results=[passing()])

    with client.websocket_connect("/ws") as session:
        bound_receives(monkeypatch, session)
        session.send_json({"goal": "ship it"})
        drain_until(session, "run.done")

    assert not server._ACTIVE_RUNS, "a finished run must be released"


# --------------------------------------------------------------------------------------
# /api/start validates and reports, but never runs
# --------------------------------------------------------------------------------------


def test_api_start_rejects_an_empty_goal(client: TestClient) -> None:
    assert client.post("/api/start", json={"goal": ""}).status_code == 422


def test_api_start_rejects_a_whitespace_goal_with_an_actionable_message(
    client: TestClient,
) -> None:
    response = client.post("/api/start", json={"goal": "   "})

    assert response.status_code == 422
    assert "goal is required" in response.json()["detail"]


def test_api_start_requires_a_goal_field(client: TestClient) -> None:
    assert client.post("/api/start", json={}).status_code == 422


def test_api_start_echoes_the_effective_configuration(client: TestClient) -> None:
    response = client.post("/api/start", json={"goal": "  ship it  "})

    assert response.status_code == 200
    body = response.json()
    assert body["goal"] == "ship it", "the goal must be reported stripped"
    assert body["provider"] == "anthropic"
    assert body["max_cycles"] == 1, "the configured cycle limit must be reported"
    assert body["max_parallel_tasks"] == 5
    assert body["websocket"] == "/ws"
    for role in ("model", "planner_model", "worker_model", "tester_model"):
        assert body[role], f"{role} must be part of the effective configuration"


def test_api_start_reports_the_workspace_it_would_use(
    client: TestClient,
    tmp_workspace: Path,
) -> None:
    response = client.post("/api/start", json={"goal": "ship it"})

    assert response.json()["workspace"] == str(tmp_workspace)


def test_api_start_applies_one_model_to_every_role(client: TestClient) -> None:
    body = client.post("/api/start", json={"goal": "ship it", "model": "house-model"}).json()

    assert body["model"] == "house-model"
    assert body["planner_model"] == "house-model"
    assert body["worker_model"] == "house-model"
    assert body["tester_model"] == "house-model"


def test_api_start_keeps_the_role_models_when_the_override_is_blank(client: TestClient) -> None:
    body = client.post("/api/start", json={"goal": "ship it", "model": "  "}).json()
    defaults = build_settings()

    assert body["planner_model"] == defaults.planner_model
    assert body["worker_model"] == defaults.worker_model
    assert body["tester_model"] == defaults.tester_model


def test_api_start_never_starts_a_run(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    goals = spy_orchestrators(monkeypatch)

    response = client.post("/api/start", json={"goal": "ship it"})

    assert response.status_code == 200
    assert goals == [], "the endpoint validates, the websocket runs"
    assert not server._ACTIVE_RUNS, "the endpoint must not start a run"


def test_api_start_does_not_serve_the_runner(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served: list[str] = []

    def fake_run(*args: Any, **kwargs: Any) -> None:
        served.append("served")

    monkeypatch.setattr(server, "uvicorn", SimpleNamespace(run=fake_run))

    client.post("/api/start", json={"goal": "ship it"})

    assert served == []


# --------------------------------------------------------------------------------------
# Binding: this endpoint has no authentication
# --------------------------------------------------------------------------------------


def test_the_default_host_is_loopback() -> None:
    assert server.DEFAULT_HOST == "127.0.0.1"
    assert ipaddress.ip_address(server.DEFAULT_HOST).is_loopback


def test_the_default_host_is_not_world_binding() -> None:
    """``0.0.0.0`` would publish an unauthenticated agent that runs commands and writes files."""
    assert server.DEFAULT_HOST not in {"0.0.0.0", "::", ""}
    assert not ipaddress.ip_address(server.DEFAULT_HOST).is_unspecified
    assert server.DEFAULT_PORT == 8000


def test_run_passes_the_bind_address_to_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(server, "uvicorn", SimpleNamespace(run=lambda *a, **k: calls.append(k)))

    server.run(host="127.0.0.1", port=1234, log_level="DEBUG")

    assert calls == [{"host": "127.0.0.1", "port": 1234, "log_level": "debug"}]


def test_run_defaults_the_host_port_and_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(server, "uvicorn", SimpleNamespace(run=lambda *a, **k: calls.append(k)))

    server.run()

    assert calls == [
        {"host": server.DEFAULT_HOST, "port": server.DEFAULT_PORT, "log_level": "info"}
    ]


def test_run_warns_when_binding_a_public_address(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(server, "uvicorn", SimpleNamespace(run=lambda *a, **k: None))

    with caplog.at_level(logging.WARNING, logger="furrow.web.server"):
        server.run(host="0.0.0.0")

    assert "without authentication" in caplog.text
    assert "0.0.0.0" in caplog.text


@pytest.mark.parametrize("host", sorted(server.LOOPBACK_HOSTS))
def test_run_does_not_warn_for_a_loopback_alias(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    host: str,
) -> None:
    monkeypatch.setattr(server, "uvicorn", SimpleNamespace(run=lambda *a, **k: None))

    with caplog.at_level(logging.WARNING, logger="furrow.web.server"):
        server.run(host=host)

    assert "without authentication" not in caplog.text


# --------------------------------------------------------------------------------------
# _send: a client that stopped reading must not take the run down
# --------------------------------------------------------------------------------------


async def test_send_builds_a_message_with_a_type_and_a_message(
    bounded: Callable[..., Awaitable[bool]],
) -> None:
    socket: Any = FakeWebSocket()

    assert await bounded(server._send(socket, "cycle.start", {"message": "Starting cycle 1"}))

    assert socket.sent == [{"type": "cycle.start", "message": "Starting cycle 1", "cycle": None}]


async def test_send_defaults_a_missing_message_and_cycle(
    bounded: Callable[..., Awaitable[bool]],
) -> None:
    socket: Any = FakeWebSocket()

    assert await bounded(server._send(socket, "ping"))

    assert socket.sent == [{"type": "ping", "message": "", "cycle": None}]


async def test_send_keeps_the_extra_keys_of_an_event(
    bounded: Callable[..., Awaitable[bool]],
) -> None:
    socket: Any = FakeWebSocket()
    payload = {"message": "Tests failed", "passed": False, "failures": ["boom"]}

    assert await bounded(server._send(socket, "test.result", payload))

    assert socket.sent[0]["passed"] is False
    assert socket.sent[0]["failures"] == ["boom"]


async def test_send_lets_the_event_name_win_over_the_payload(
    bounded: Callable[..., Awaitable[bool]],
) -> None:
    socket: Any = FakeWebSocket()

    assert await bounded(server._send(socket, "run.done", {"type": "spoofed", "message": "m"}))

    assert socket.sent[0]["type"] == "run.done"


@pytest.mark.parametrize(
    "failure",
    [WebSocketDisconnect(1001), RuntimeError("already closed"), OSError("broken pipe")],
    ids=["disconnect", "closed", "oserror"],
)
async def test_send_returns_false_when_the_client_is_gone(
    bounded: Callable[..., Awaitable[bool]],
    failure: BaseException,
) -> None:
    """A send failure ends the stream quietly: the orchestrator awaits this on every event."""
    socket: Any = FakeWebSocket(failure=failure)

    assert await bounded(server._send(socket, "run.start", {"message": "x"})) is False


async def test_a_transient_send_failure_only_costs_the_next_send(
    bounded: Callable[..., Awaitable[bool]],
) -> None:
    """Recovering is allowed: one failed event must not wedge the rest of the run."""
    socket = FakeWebSocket(failure=OSError("broken pipe"))
    gone: Any = socket

    assert await bounded(server._send(gone, "run.start", {"message": "x"})) is False

    socket.failure = None
    assert await bounded(server._send(gone, "cycle.start", {"message": "y"})) is True
    assert socket.sent == [{"type": "cycle.start", "message": "y", "cycle": None}]


async def test_the_stream_handler_forwards_every_event(
    bounded: Callable[..., Awaitable[None]],
) -> None:
    socket = FakeWebSocket()
    handler = server._stream_handler(socket)  # type: ignore[arg-type]

    await bounded(handler("cycle.done", {"message": "Cycle 1 finished", "cycle": 1}))

    assert socket.sent[0]["type"] == "cycle.done"
    assert socket.sent[0]["cycle"] == 1


# --------------------------------------------------------------------------------------
# _drive and _reap
# --------------------------------------------------------------------------------------


class ExplodingOrchestrator:
    async def run(self) -> LoopResult:
        raise RuntimeError("the loop fell over")


class CancellingOrchestrator:
    async def run(self) -> LoopResult:
        raise asyncio.CancelledError


class FinishingOrchestrator:
    result = LoopResult(goal="g", cycles=1, stopped_reason="goal-complete", passed=True)

    async def run(self) -> LoopResult:
        return self.result


async def test_drive_reports_a_crash_as_an_error_event(
    bounded: Callable[..., Awaitable[LoopResult]],
) -> None:
    socket = FakeWebSocket()
    gone: Any = socket

    with pytest.raises(RuntimeError, match="the loop fell over"):
        await bounded(server._drive(ExplodingOrchestrator(), gone))  # type: ignore[arg-type]

    assert socket.sent[0]["type"] == "error"
    assert "Furrow run crashed" in socket.sent[0]["message"]
    assert "the loop fell over" in socket.sent[0]["message"]


async def test_drive_lets_cancellation_through(
    bounded: Callable[..., Awaitable[LoopResult]],
) -> None:
    """Cancellation is control flow, not a crash: it must not be reported as an error."""
    socket = FakeWebSocket()
    gone: Any = socket

    with pytest.raises(asyncio.CancelledError):
        await bounded(server._drive(CancellingOrchestrator(), gone))  # type: ignore[arg-type]

    assert socket.sent == []


async def test_drive_returns_the_loop_result(
    bounded: Callable[..., Awaitable[LoopResult]],
) -> None:
    socket: Any = FakeWebSocket()

    assert await bounded(server._drive(FinishingOrchestrator(), socket)) is (  # type: ignore[arg-type]
        FinishingOrchestrator.result
    )
    assert socket.sent == []


async def _return(value: LoopResult) -> LoopResult:
    return value


async def _raise(error: BaseException) -> None:
    raise error


async def test_reap_cancels_a_still_running_task_and_waits_for_it(
    bounded: Callable[..., Awaitable[None]],
) -> None:
    task: Any = asyncio.create_task(asyncio.Event().wait())
    await asyncio.sleep(0)

    await bounded(server._reap(task))

    assert task.done(), "a leaked task would keep an agent billing credits"
    assert task.cancelled()


async def test_reap_leaves_a_finished_task_alone(
    bounded: Callable[..., Awaitable[None]],
) -> None:
    result = LoopResult(goal="g", cycles=1, stopped_reason="goal-complete", passed=True)
    task: Any = asyncio.create_task(_return(result))
    await asyncio.sleep(0)

    await bounded(server._reap(task))

    assert not task.cancelled()
    assert task.result() is result


async def test_reap_accepts_no_task_at_all(bounded: Callable[..., Awaitable[None]]) -> None:
    await bounded(server._reap(None))


async def test_reap_swallows_a_crashed_run(
    bounded: Callable[..., Awaitable[None]],
) -> None:
    """A crash must not become an "exception was never retrieved" warning in the server log."""
    task: Any = asyncio.create_task(_raise(RuntimeError("boom")))
    await asyncio.sleep(0)

    await bounded(server._reap(task))

    assert isinstance(task.exception(), RuntimeError)


# --------------------------------------------------------------------------------------
# _settings_for
# --------------------------------------------------------------------------------------


def test_settings_for_applies_one_model_to_every_role() -> None:
    built = server._settings_for("house-model")

    assert built.model == "house-model"
    assert built.planner_model == "house-model"
    assert built.worker_model == "house-model"
    assert built.tester_model == "house-model"


def test_settings_for_ignores_a_blank_model() -> None:
    assert server._settings_for("   ").planner_model == build_settings().planner_model


def test_settings_for_ignores_a_none_model() -> None:
    assert server._settings_for(None).planner_model == build_settings().planner_model


def test_settings_for_honours_role_models_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FURROW_WORKER_MODEL", "env-worker")

    assert server._settings_for(None).worker_model == "env-worker"


# --------------------------------------------------------------------------------------
# Per-run settings isolation: one request's override must not become the next run's default
# --------------------------------------------------------------------------------------
#
# ``Settings`` is built per request, never taken from the process-wide ``get_settings`` cache,
# because that cache is *rebuilt* by ``get_settings(**overrides)``: a single request carrying a
# model would otherwise silently become the server-wide default for every later run.


def test_settings_for_hands_every_run_its_own_instance() -> None:
    """Two runs must not end up holding the same mutable :class:`Settings` object."""
    first = server._settings_for(None)
    second = server._settings_for(None)

    assert first is not second, "a shared instance lets one run reconfigure another"


def test_settings_for_a_later_override_does_not_rewrite_an_earlier_run() -> None:
    """Both directions of the same leak: an override must not travel in either time direction."""
    defaults = build_settings()

    first = server._settings_for(None)
    second = server._settings_for("override-1")

    assert first is not second
    assert first.model == defaults.model, "run 1 must keep the model it was configured with"
    assert first.planner_model == defaults.planner_model
    assert first.worker_model == defaults.worker_model
    assert first.tester_model == defaults.tester_model
    assert second.model == "override-1"
    assert second.planner_model == "override-1"
    assert second.worker_model == "override-1"
    assert second.tester_model == "override-1"
    assert server._settings_for(None).planner_model == defaults.planner_model, (
        "the override above must not have become the default"
    )


def test_api_start_an_override_does_not_leak_into_the_next_request(client: TestClient) -> None:
    """The regression: a start request with a ``model``, then one without.

    The second request has to report the configured defaults for all four model fields. Before
    the fix it reported ``override-1``, because ``get_settings(**overrides)`` re-cached the
    override and the override-free request was served straight from that cache.
    """
    defaults = build_settings()

    first = client.post("/api/start", json={"goal": "a", "model": "override-1"}).json()
    second = client.post("/api/start", json={"goal": "b"}).json()

    assert first["model"] == "override-1", "the override must still reach the run that asked for it"
    for role in ("model", "planner_model", "worker_model", "tester_model"):
        assert second[role] == getattr(defaults, role), f"run 2 must report the default {role}"


def test_api_start_reports_the_ambient_environment(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings are built per request, so ``FURROW_*`` is still what configures a run."""
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "3")

    body = client.post("/api/start", json={"goal": "ship it"}).json()

    assert body["max_parallel_tasks"] == 3


def test_api_start_reads_the_environment_after_an_earlier_override(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An earlier request must not freeze the configuration the later ones see."""
    client.post("/api/start", json={"goal": "a", "model": "override-1"})
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "7")

    body = client.post("/api/start", json={"goal": "b"}).json()

    assert body["max_parallel_tasks"] == 7


def test_ws_a_model_override_on_one_run_does_not_reach_the_next_run(
    stub_agents: StubInstaller,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same leak over ``/ws``, where it decides which model actually bills the run.

    Run 1 asks for one model and run 2 asks for none; run 2 must reach its ``LLMClient`` with the
    configured role models rather than with run 1's override.
    """
    stub_agents(plans=[make_plan("x")], test_results=[passing()])
    clients = record_llm_clients(monkeypatch)
    defaults = build_settings()

    def run_a_websocket(payload: dict[str, Any]) -> None:
        # A TestClient per run: each owns its own portal thread, and the deadline wrapper of a
        # finished session cannot be left wrapping the portal the next session would reuse.
        with TestClient(server.app) as test_client, test_client.websocket_connect("/ws") as session:
            bound_receives(monkeypatch, session)
            session.send_json(payload)
            drain_until(session, "run.done")

    run_a_websocket({"goal": "a", "model": "override-1"})
    run_a_websocket({"goal": "b"})

    assert len(clients) == 2, "both runs must have built a client"
    first, second = (recorded.settings for recorded in clients)
    assert first is not second
    assert first.tester_model == "override-1"
    assert second.model == defaults.model
    assert second.planner_model == defaults.planner_model
    assert second.worker_model == defaults.worker_model
    assert second.tester_model == defaults.tester_model
