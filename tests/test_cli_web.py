"""Offline tests for the furrow command line entry point and the web front end.

The orchestrator and the LLM client are replaced by stubs, so the CLI and the HTTP
endpoints are exercised end to end without a network call, an API key or a real run:
``CliRunner`` drives the click group and ``fastapi.testclient.TestClient`` drives the
ASGI app in-process.  No test here starts a listening socket -- ``run()`` is only ever
called with ``uvicorn`` stubbed out.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import importlib
import io
import json
import os
import signal
import threading
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from click.testing import CliRunner, Result
from fastapi.testclient import TestClient
from rich.console import Console
from starlette.websockets import WebSocketDisconnect

from furrow import __version__
from furrow.config import Provider, RunResult, Settings
from furrow.config import settings as global_settings
from furrow.core.events import EventCallback, EventKind, make_event
from furrow.web import server

# ``furrow.cli`` re-exports the click entry point as ``main``, which shadows the submodule
# attribute of the same name, so the module has to be fetched explicitly: the tests below
# patch module level names (``console``, ``Orchestrator``, ``LLMClient``).
cli_main = importlib.import_module("furrow.cli.main")

GOAL = "ship the widget factory"

# Leaving a websocket session mid-run makes starlette's test client cancel the future
# backing the ASGI task, so the cancellation shows up as one of these, not as an assert.
DISCONNECT_NOISE = (asyncio.CancelledError, concurrent.futures.CancelledError)


# ---------------------------------------------------------------------------
# stubs: nothing below ever touches a provider
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """The smallest object the CLI and the web server accept as an LLM client."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Store the settings; no provider client is ever built."""
        self.settings = settings
        self.closed = False

    async def aclose(self) -> None:
        """Record that the client was closed."""
        self.closed = True


class CliStubs:
    """The script the stubbed CLI orchestrator follows, plus the calls it received."""

    def __init__(self) -> None:
        """Default to a single successful cycle."""
        self.outcome: Any = RunResult(
            goal=GOAL, cycles=1, completed=True, reason="goal complete", tests_passed=True
        )
        self.error: BaseException | None = None
        self.sigint_on_run = False
        self.orchestrators: list[StubCliOrchestrator] = []
        self.clients: list[Settings] = []

    def factory(self, goal: str, **kwargs: Any) -> StubCliOrchestrator:
        """Build the stub orchestrator the CLI will drive."""
        orchestrator = StubCliOrchestrator(goal, self, **kwargs)
        self.orchestrators.append(orchestrator)
        return orchestrator

    def client_factory(self, settings: Settings | None = None) -> FakeLLMClient:
        """Build a fake client and remember the settings it was given."""
        self.clients.append(settings)
        return FakeLLMClient(settings)


class StubCliOrchestrator:
    """Records the kwargs the CLI passes and replays a canned outcome."""

    def __init__(self, goal: str, stubs: CliStubs, **kwargs: Any) -> None:
        """Store the goal, the forwarded kwargs and the shared stubs."""
        self.goal = goal
        self.kwargs = kwargs
        self.stubs = stubs
        self.client = kwargs.get("client")
        self.settings = kwargs.get("settings")
        self.on_event: EventCallback | None = kwargs.get("on_event")
        self.stop_calls = 0
        self.run_calls = 0

    async def run(self) -> RunResult:
        """Return the canned result, or raise the canned error."""
        self.run_calls += 1
        if self.stubs.sigint_on_run:
            os.kill(os.getpid(), signal.SIGINT)
            await asyncio.sleep(0.05)  # let the loop deliver the stop request
        if self.stubs.error is not None:
            raise self.stubs.error
        return self.stubs.outcome

    async def stop(self) -> None:
        """Record a cooperative stop request."""
        self.stop_calls += 1


class WebStubs:
    """The script the stubbed web orchestrator follows, plus the calls it received."""

    def __init__(self) -> None:
        """Default to a short successful run emitting one event per kind."""
        self.completed = True
        self.reason = "goal complete"
        self.tests_passed = True
        self.tasks_completed = 1
        self.hangs = False
        self.error: BaseException | None = None
        self.events: list[tuple[EventKind, str, dict[str, Any]]] = [
            (EventKind.CYCLE, "cycle 1 started", {}),
            (EventKind.TASK_DONE, "wrote widget.py", {"files": 1}),
            (EventKind.TESTS, "1 passed", {"passed": True, "summary": "1 passed"}),
            (EventKind.DONE, "goal complete", {"completed": True, "reason": "goal complete"}),
        ]
        self.instances: list[StubWebOrchestrator] = []
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.cancelled = threading.Event()

    def factory(self, goal: str, **kwargs: Any) -> StubWebOrchestrator:
        """Build the stub orchestrator the server will drive."""
        return StubWebOrchestrator(goal, self, **kwargs)


class StubWebOrchestrator:
    """Emits a scripted event stream through ``on_event`` and returns a run result."""

    def __init__(self, goal: str, stubs: WebStubs, **kwargs: Any) -> None:
        """Store the goal, the forwarded kwargs and the shared stubs."""
        self.goal = goal
        self.kwargs = kwargs
        self.stubs = stubs
        self.settings = kwargs.get("settings")
        self.client = kwargs.get("client")
        self.on_event = kwargs.get("on_event")
        self.stop_calls = 0

    async def run(self) -> RunResult:
        """Emit the scripted events, honouring a scripted hang or failure."""
        stubs = self.stubs
        stubs.instances.append(self)
        stubs.started.set()
        if stubs.hangs:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                stubs.cancelled.set()
                raise
        if stubs.error is not None:
            raise stubs.error
        for kind, message, data in stubs.events:
            await self.on_event(make_event(kind, message, cycle=1, data=data))
        return RunResult(
            goal=self.goal,
            cycles=1,
            completed=stubs.completed,
            reason=stubs.reason,
            tests_passed=stubs.tests_passed,
            tasks_completed=stubs.tasks_completed,
        )

    async def stop(self) -> None:
        """Record a cooperative stop request."""
        self.stop_calls += 1
        self.stubs.stopped.set()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """Return a click test runner."""
    return CliRunner()


@pytest.fixture
def workspace(tmp_path: Any) -> Any:
    """Return a throwaway workspace directory."""
    return tmp_path


@pytest.fixture
def cli_output(monkeypatch: pytest.MonkeyPatch) -> Callable[[], str]:
    """Redirect the CLI's rich console into a buffer and return a reader for it."""
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, no_color=True, highlight=False)
    monkeypatch.setattr(cli_main, "console", console)
    return buffer.getvalue


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> CliStubs:
    """Replace the CLI's orchestrator and LLM client with recording stubs."""
    stubs = CliStubs()
    monkeypatch.setattr(cli_main, "Orchestrator", stubs.factory)
    monkeypatch.setattr(cli_main, "LLMClient", stubs.client_factory)
    return stubs


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch, workspace: Any) -> WebStubs:
    """Replace the web server's orchestrator and client, and give it a fake key."""
    stubs = WebStubs()
    monkeypatch.setattr(server, "Orchestrator", stubs.factory)
    monkeypatch.setattr(server, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(
        server,
        "settings",
        Settings(workspace=workspace, anthropic_api_key="sk-do-not-leak"),
    )
    return stubs


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Return an in-process client for the web app."""
    with TestClient(server.app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# the group itself
# ---------------------------------------------------------------------------


def test_version_flag_prints_the_package_version(runner: CliRunner) -> None:
    """``furrow --version`` prints the version of the installed package."""
    result: Result = runner.invoke(cli_main.main, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"furrow {__version__}"


def test_help_lists_both_commands(runner: CliRunner) -> None:
    """The group advertises the two commands it has."""
    result: Result = runner.invoke(cli_main.main, ["--help"])

    assert result.exit_code == 0
    assert "start" in result.output
    assert "web" in result.output


# ---------------------------------------------------------------------------
# clean validation instead of tracebacks
# ---------------------------------------------------------------------------


def test_an_empty_goal_is_a_click_validation_error(runner: CliRunner) -> None:
    """``start ""`` fails with exit code 2 and a message, never a traceback."""
    result: Result = runner.invoke(cli_main.main, ["start", ""])

    assert result.exit_code == 2
    assert "goal must not be empty" in result.output
    assert "Traceback" not in result.output


def test_a_missing_workspace_directory_is_rejected(runner: CliRunner, tmp_path: Any) -> None:
    """A ``--workspace`` that does not exist is a bad parameter, not a crash."""
    missing = tmp_path / "nope"
    result: Result = runner.invoke(cli_main.main, ["start", GOAL, "--workspace", str(missing)])

    assert result.exit_code == 2
    assert "workspace directory does not exist" in result.output
    assert "--workspace" in result.output
    assert "Traceback" not in result.output


def test_a_negative_max_cycles_is_rejected(runner: CliRunner, workspace: Any) -> None:
    """A negative cycle limit is rejected before any run starts."""
    result: Result = runner.invoke(
        cli_main.main,
        ["start", GOAL, "--workspace", str(workspace), "--max-cycles", "-1"],
    )

    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_a_zero_max_parallel_is_rejected(runner: CliRunner, workspace: Any) -> None:
    """``--max-parallel`` below one is a bad parameter."""
    result: Result = runner.invoke(
        cli_main.main,
        ["start", GOAL, "--workspace", str(workspace), "--max-parallel", "0"],
    )

    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_an_unknown_provider_is_rejected(runner: CliRunner, workspace: Any) -> None:
    """``--provider`` only accepts the known provider names."""
    result: Result = runner.invoke(
        cli_main.main,
        ["start", GOAL, "--workspace", str(workspace), "--provider", "nosuch"],
    )

    assert result.exit_code == 2
    assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# settings wiring
# ---------------------------------------------------------------------------


def test_start_builds_a_fresh_settings_object(
    runner: CliRunner,
    cli: CliStubs,
    workspace: Any,
) -> None:
    """The run gets its own ``Settings``, never the process-wide singleton."""
    before = global_settings.model_dump()

    result = runner.invoke(
        cli_main.main,
        ["start", GOAL, "--workspace", str(workspace), "--max-cycles", "4"],
    )

    assert result.exit_code == 0
    assert cli.orchestrators[0].settings is not global_settings
    assert global_settings.model_dump() == before


def test_the_client_and_the_orchestrator_share_one_settings_object(
    runner: CliRunner,
    cli: CliStubs,
    workspace: Any,
) -> None:
    """``LLMClient`` and ``Orchestrator`` are built from the same settings instance."""
    result = runner.invoke(
        cli_main.main,
        ["start", GOAL, "--workspace", str(workspace)],
    )

    assert result.exit_code == 0
    assert cli.clients == [cli.orchestrators[0].settings]
    assert isinstance(cli.orchestrators[0].client, FakeLLMClient)


def test_start_passes_the_workspace_through(
    runner: CliRunner,
    cli: CliStubs,
    workspace: Any,
) -> None:
    """``--workspace`` reaches the settings handed to the orchestrator."""
    result = runner.invoke(cli_main.main, ["start", GOAL, "--workspace", str(workspace)])

    assert result.exit_code == 0
    assert cli.orchestrators[0].settings.workspace == workspace.resolve()


def test_model_propagates_to_every_role(
    runner: CliRunner,
    cli: CliStubs,
    workspace: Any,
) -> None:
    """``--model`` sets the planner, worker and tester models too."""
    result = runner.invoke(
        cli_main.main,
        ["start", GOAL, "--workspace", str(workspace), "--model", "shared-model"],
    )

    assert result.exit_code == 0
    cfg = cli.orchestrators[0].settings
    assert cfg.model == "shared-model"
    assert cfg.planner_model == "shared-model"
    assert cfg.worker_model == "shared-model"
    assert cfg.tester_model == "shared-model"


def test_an_explicit_role_model_wins_over_model(
    runner: CliRunner,
    cli: CliStubs,
    workspace: Any,
) -> None:
    """``--worker-model`` overrides the model ``--model`` applied to every role."""
    result = runner.invoke(
        cli_main.main,
        [
            "start",
            GOAL,
            "--workspace",
            str(workspace),
            "--model",
            "shared-model",
            "--worker-model",
            "worker-only",
        ],
    )

    assert result.exit_code == 0
    cfg = cli.orchestrators[0].settings
    assert cfg.worker_model == "worker-only"
    assert cfg.model == "shared-model"
    assert cfg.planner_model == "shared-model"
    assert cfg.tester_model == "shared-model"


def test_max_cycles_and_max_parallel_reach_the_settings(
    runner: CliRunner,
    cli: CliStubs,
    workspace: Any,
) -> None:
    """The two numeric options map onto their settings fields."""
    result = runner.invoke(
        cli_main.main,
        [
            "start",
            GOAL,
            "--workspace",
            str(workspace),
            "--max-cycles",
            "4",
            "--max-parallel",
            "3",
        ],
    )

    assert result.exit_code == 0
    cfg = cli.orchestrators[0].settings
    assert cfg.max_cycles == 4
    assert cfg.max_parallel_tasks == 3


def test_the_provider_option_reaches_the_settings(
    runner: CliRunner,
    cli: CliStubs,
    workspace: Any,
) -> None:
    """``--provider`` selects the configured provider."""
    result = runner.invoke(
        cli_main.main,
        ["start", GOAL, "--workspace", str(workspace), "--provider", "ollama"],
    )

    assert result.exit_code == 0
    assert cli.orchestrators[0].settings.provider is Provider.OLLAMA


def test_the_goal_reaches_the_orchestrator_trimmed(
    runner: CliRunner,
    cli: CliStubs,
    workspace: Any,
) -> None:
    """The goal is passed on without surrounding whitespace."""
    result = runner.invoke(
        cli_main.main,
        ["start", f"  {GOAL}  ", "--workspace", str(workspace)],
    )

    assert result.exit_code == 0
    assert cli.orchestrators[0].goal == GOAL


def test_verbose_is_forwarded_to_the_settings_and_the_printer(
    runner: CliRunner,
    cli: CliStubs,
    workspace: Any,
) -> None:
    """``--verbose`` reaches both the settings and the event printer."""
    result = runner.invoke(
        cli_main.main,
        ["start", GOAL, "--workspace", str(workspace), "--verbose"],
    )

    assert result.exit_code == 0
    assert cli.orchestrators[0].settings.verbose is True
    assert cli.orchestrators[0].on_event is not None


# ---------------------------------------------------------------------------
# exit codes
# ---------------------------------------------------------------------------


def test_a_completed_run_exits_zero(
    runner: CliRunner,
    cli: CliStubs,
    cli_output: Callable[[], str],
    workspace: Any,
) -> None:
    """``completed=True`` is a successful process."""
    cli.outcome = RunResult(goal=GOAL, cycles=1, completed=True, reason="goal complete")

    result = runner.invoke(cli_main.main, ["start", GOAL, "--workspace", str(workspace)])

    assert result.exit_code == 0
    assert "Run summary" in cli_output()


def test_an_incomplete_run_exits_one(
    runner: CliRunner,
    cli: CliStubs,
    cli_output: Callable[[], str],
    workspace: Any,
) -> None:
    """``completed=False`` is a failed process."""
    cli.outcome = RunResult(goal=GOAL, cycles=2, completed=False, reason="no progress")

    result = runner.invoke(cli_main.main, ["start", GOAL, "--workspace", str(workspace)])

    assert result.exit_code == 1
    assert "incomplete" in cli_output()
    assert "Traceback" not in cli_output()


def test_a_keyboard_interrupt_exits_130(
    runner: CliRunner,
    cli: CliStubs,
    workspace: Any,
) -> None:
    """A Ctrl-C reaching the CLI is exit code 130."""
    cli.error = KeyboardInterrupt()

    result = runner.invoke(cli_main.main, ["start", GOAL, "--workspace", str(workspace)])

    assert result.exit_code == 130


def test_a_keyboard_interrupt_prints_the_stop_message(
    runner: CliRunner,
    cli: CliStubs,
    cli_output: Callable[[], str],
    workspace: Any,
) -> None:
    """The interrupt path says the run stopped, without a traceback."""
    cli.error = KeyboardInterrupt()

    runner.invoke(cli_main.main, ["start", GOAL, "--workspace", str(workspace)])

    printed = cli_output()
    assert "Furrow stopped by user." in printed
    assert "Traceback" not in printed


def test_an_ordinary_failure_is_reported_and_exits_one(
    runner: CliRunner,
    cli: CliStubs,
    cli_output: Callable[[], str],
    workspace: Any,
) -> None:
    """An unexpected error becomes a one line message, not a traceback."""
    cli.error = RuntimeError("provider exploded")

    result = runner.invoke(cli_main.main, ["start", GOAL, "--workspace", str(workspace)])

    assert result.exit_code == 1
    printed = cli_output()
    assert "Run failed" in printed
    assert "provider exploded" in printed
    assert "Traceback" not in printed


def test_sigint_becomes_a_cooperative_stop_and_exit_130(
    runner: CliRunner,
    cli: CliStubs,
    cli_output: Callable[[], str],
    workspace: Any,
) -> None:
    """A real SIGINT stops the run at the next checkpoint instead of unwinding."""
    cli.sigint_on_run = True
    cli.outcome = RunResult(goal=GOAL, cycles=1, completed=False, reason="stop requested")

    result = runner.invoke(cli_main.main, ["start", GOAL, "--workspace", str(workspace)])

    assert result.exit_code == 130
    assert cli.orchestrators[0].stop_calls >= 1
    printed = cli_output()
    assert "Furrow stopped by user." in printed
    assert "Traceback" not in printed


# ---------------------------------------------------------------------------
# the web command
# ---------------------------------------------------------------------------


def test_web_help_renders(runner: CliRunner) -> None:
    """``furrow web --help`` documents the binding options."""
    result: Result = runner.invoke(cli_main.main, ["web", "--help"])

    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output


def test_web_forwards_host_and_port_to_run(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--host`` and ``--port`` are handed to ``furrow.web.server.run``."""
    captured: dict[str, Any] = {}

    def fake_run(host: str = "127.0.0.1", port: int = 8000, *, reload: bool = False) -> None:
        captured.update(host=host, port=port, reload=reload)

    monkeypatch.setattr(server, "run", fake_run)

    result = runner.invoke(cli_main.main, ["web", "--host", "0.0.0.0", "--port", "9111"])

    assert result.exit_code == 0
    assert captured == {"host": "0.0.0.0", "port": 9111, "reload": False}


def test_web_defaults_to_loopback_on_8000(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without options the server binds loopback on port 8000."""
    captured: dict[str, Any] = {}

    def fake_run(host: str = "127.0.0.1", port: int = 8000, *, reload: bool = False) -> None:
        captured.update(host=host, port=port)

    monkeypatch.setattr(server, "run", fake_run)

    result = runner.invoke(cli_main.main, ["web"])

    assert result.exit_code == 0
    assert captured == {"host": "127.0.0.1", "port": 8000}


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


def test_index_serves_the_run_form_and_its_script(client: TestClient) -> None:
    """``GET /`` returns the goal input, the model input and the driving script."""
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert 'id="goal"' in body
    assert 'id="model"' in body
    assert "<script" in body
    assert "/ws" in body


def test_health_reports_ok_and_the_version(client: TestClient) -> None:
    """``GET /api/health`` is the probe endpoint."""
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == __version__


def test_config_reports_the_provider_and_the_role_models(client: TestClient, web: WebStubs) -> None:
    """``GET /api/config`` describes the effective models."""
    response = client.get("/api/config")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "provider",
        "model",
        "planner_model",
        "worker_model",
        "tester_model",
    }
    assert payload["provider"] in {provider.value for provider in Provider}
    assert all(isinstance(value, str) and value for value in payload.values())


def test_config_leaks_no_api_key(client: TestClient, web: WebStubs) -> None:
    """Even with a key configured, ``/api/config`` exposes no credential."""
    body = client.get("/api/config").text

    assert "sk-" not in body
    assert "sk-do-not-leak" not in body
    assert "api_key" not in body
    assert "anthropic_api_key" not in body


# ---------------------------------------------------------------------------
# the websocket run stream
# ---------------------------------------------------------------------------


def test_websocket_streams_the_events_in_order_ending_with_done(
    client: TestClient,
    web: WebStubs,
) -> None:
    """The client receives one event per frame, the last one ``done``."""
    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": GOAL})
        payloads = [socket.receive_json() for _ in range(len(web.events))]

    assert [payload["kind"] for payload in payloads] == [
        EventKind.CYCLE.value,
        EventKind.TASK_DONE.value,
        EventKind.TESTS.value,
        EventKind.DONE.value,
    ]
    assert payloads[-1]["data"]["reason"] == "goal complete"
    assert payloads[1]["data"]["files"] == 1


def test_websocket_payloads_are_json_lines(client: TestClient, web: WebStubs) -> None:
    """Every streamed frame is one JSON object carrying the full event shape."""
    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": GOAL})
        frames = [socket.receive_text() for _ in range(len(web.events))]

    for frame in frames:
        assert "\n" not in frame
        payload = json.loads(frame)
        assert set(payload) == {"kind", "message", "cycle", "task_id", "data", "ts"}
        assert payload["cycle"] == 1


def test_websocket_closes_after_the_run_finishes(client: TestClient, web: WebStubs) -> None:
    """The server closes the socket, so the browser's ``onclose`` fires."""
    with client.websocket_connect("/ws") as socket, pytest.raises(WebSocketDisconnect):
        socket.send_json({"goal": GOAL})
        while socket.receive_json()["kind"] != EventKind.DONE.value:
            pass
        socket.receive_json()  # the server closed once the run ended, so this raises


def test_websocket_hands_the_goal_to_the_orchestrator(client: TestClient, web: WebStubs) -> None:
    """The orchestrator is built with the goal the client sent."""
    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": GOAL})
        socket.receive_json()

    assert web.instances
    assert web.instances[0].goal == GOAL


def test_a_requested_model_applies_to_every_role(client: TestClient, web: WebStubs) -> None:
    """``{"model": ...}`` overrides the model of every role, like ``--model``."""
    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": GOAL, "model": "picked-model"})
        socket.receive_json()

    cfg = web.instances[0].settings
    assert cfg.model == "picked-model"
    assert cfg.planner_model == "picked-model"
    assert cfg.worker_model == "picked-model"
    assert cfg.tester_model == "picked-model"


def test_a_payload_without_a_goal_yields_an_error_event(client: TestClient, web: WebStubs) -> None:
    """``{}`` is rejected with an ``error`` event instead of starting a run."""
    with client.websocket_connect("/ws") as socket:
        socket.send_json({})
        payload = socket.receive_json()

    assert payload["kind"] == EventKind.ERROR.value
    assert "goal" in payload["message"]
    assert web.instances == []


def test_a_blank_goal_yields_an_error_event(client: TestClient, web: WebStubs) -> None:
    """A whitespace-only goal is rejected before a run is started."""
    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": "   "})
        payload = socket.receive_json()

    assert payload["kind"] == EventKind.ERROR.value
    assert payload["message"] == "goal must not be empty"
    assert web.instances == []


def test_invalid_json_yields_an_error_event(client: TestClient, web: WebStubs) -> None:
    """A frame that is not JSON at all is reported, not raised."""
    with client.websocket_connect("/ws") as socket:
        socket.send_text("{not json")
        payload = socket.receive_json()

    assert payload["kind"] == EventKind.ERROR.value
    assert "invalid payload" in payload["message"]
    assert web.instances == []


def test_a_failing_run_reports_the_failure_as_an_error_event(
    client: TestClient,
    web: WebStubs,
) -> None:
    """An exception out of the run becomes an ``error`` event on the socket."""
    web.error = RuntimeError("run blew up")

    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": GOAL})
        payload = socket.receive_json()

    assert payload["kind"] == EventKind.ERROR.value
    assert "run blew up" in payload["message"]


def test_an_early_disconnect_asks_the_orchestrator_to_stop(
    client: TestClient,
    web: WebStubs,
) -> None:
    """The disconnect path calls ``stop()`` instead of leaking the run task."""
    web.hangs = True

    with contextlib.suppress(*DISCONNECT_NOISE):
        with client.websocket_connect("/ws") as socket:
            socket.send_json({"goal": GOAL})
            assert web.started.wait(timeout=10)
            socket.close(1000)
            assert web.stopped.wait(timeout=10)

    assert web.instances
    assert web.instances[0].stop_calls == 1


def test_an_early_disconnect_cancels_the_run_task(client: TestClient, web: WebStubs) -> None:
    """After the cooperative stop the orchestrator task is cancelled."""
    web.hangs = True

    with contextlib.suppress(*DISCONNECT_NOISE):
        with client.websocket_connect("/ws") as socket:
            socket.send_json({"goal": GOAL})
            assert web.started.wait(timeout=10)
            socket.close(1000)
            assert web.cancelled.wait(timeout=10)

    assert web.cancelled.is_set()


# ---------------------------------------------------------------------------
# the uvicorn launcher
# ---------------------------------------------------------------------------


def test_run_rejects_port_zero() -> None:
    """``port=0`` is refused before anything is bound."""
    with pytest.raises(ValueError, match="port must be between 1 and 65535"):
        server.run(port=0)


def test_run_rejects_an_out_of_range_port() -> None:
    """Ports outside 1..65535 are refused."""
    with pytest.raises(ValueError, match="port must be between 1 and 65535"):
        server.run(port=70000)


def test_run_hands_the_arguments_to_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid port is forwarded to uvicorn together with the app, host and port."""
    captured: dict[str, Any] = {}

    class FakeUvicorn:
        """Stands in for the uvicorn module so no server is ever started."""

        @staticmethod
        def run(app: Any, **kwargs: Any) -> None:
            captured["app"] = app
            captured.update(kwargs)

    monkeypatch.setattr(server, "uvicorn", FakeUvicorn)

    server.run(host="0.0.0.0", port=9123)

    assert captured == {"app": server.app, "host": "0.0.0.0", "port": 9123, "reload": False}