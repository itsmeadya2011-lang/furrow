"""Tests for the two user-facing surfaces: the ``furrow`` CLI and the web layer.

Both surfaces drive a real :class:`~furrow.core.orchestrator.Orchestrator` in
production, so nothing here talks to an LLM: the CLI tests swap
``furrow.cli.main.Orchestrator`` for a *recording* fake (which doubles as the
proof that no orchestrator was built at all), and the websocket tests swap
``furrow.web.server.Orchestrator`` for an *event-scripting* fake that replays a
known orchestrator event stream.

Three properties are load-bearing and are pinned deliberately:

* ``furrow start`` builds a **new** :class:`~furrow.config.Settings` from the
  command line. The process-wide ``settings`` singleton must come out of an
  invocation untouched, or repeated in-process runs (and the web server, which
  reads the same singleton) inherit stale overrides.
* ``/ws`` forwards **one JSON message per orchestrator event**. The web UI used
  to open a socket, never send a single frame, and leave the page on
  "Starting..." forever, so the streaming, the payload shape and the terminal
  ``run_finished`` message are all asserted here.
* The web layer allows exactly one run at a time and keeps it in a module-level
  registry. Two real bugs came out of that registry: a rejected second start
  still built an ``Orchestrator``, and a handler cancelled by a vanishing client
  never released the slot, wedging the server for good. Both are regression
  tests here, and an autouse fixture empties the slot before and after every test
  in the module so one leak cannot cascade into the rest of the suite.

The websocket is driven through :class:`fastapi.testclient.TestClient`, i.e.
Starlette's in-process ASGI transport: no listening socket, no ``uvicorn.run``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

import furrow.cli.main as cli_module
import furrow.web.server as web
from furrow.cli.main import ConfigError
from furrow.config import Provider, Settings, settings

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def wide_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop rich from hard-wrapping CLI output at 80 columns.

    The CLI prints through ``rich.console.Console``, which takes its width from
    ``COLUMNS``. Without this a long error message would be split across lines
    and substring assertions on it would depend on the terminal the suite
    happens to run in.
    """
    monkeypatch.setenv("COLUMNS", "200")


@pytest.fixture(autouse=True)
def clean_run_registry() -> Iterator[None]:
    """Keep the web layer's process-wide active-run slot empty around each test.

    ``furrow.web.server`` keeps the single permitted run in a module-level
    registry guarded by a lock. A leaked entry makes every later websocket test
    fail with ``run_in_progress``, so the slot is emptied before *and* after each
    test even if the test failed half way through. Tests that start a run still
    drive it to its natural end -- this fixture is the safety net, not the
    mechanism.
    """
    _clear_run_registry()
    yield
    _clear_run_registry()


@pytest.fixture(autouse=True)
def bounded_shutdown_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the web layer's stop grace period so a stuck run fails fast.

    ``furrow.web.server`` waits ``SHUTDOWN_TIMEOUT`` seconds for a stopped run to
    unwind before cancelling it. No test here should ever need that budget, and
    a regression that does should surface in seconds rather than half a minute.
    """
    monkeypatch.setattr(web, "SHUTDOWN_TIMEOUT", 2.0)


@pytest.fixture
def web_client() -> Iterator[TestClient]:
    """A ``TestClient`` over the ASGI app -- in-process transport, no sockets."""
    client = TestClient(web.app)
    yield client
    client.close()


# --------------------------------------------------------------------------- #
# Helpers: CLI
# --------------------------------------------------------------------------- #


def _run_cli(*args: str, stdin: str | None = None) -> Result:
    """Invoke the ``furrow`` CLI group in-process and capture the result."""
    return CliRunner().invoke(cli_module.main, list(args), input=stdin)


@dataclass
class _RecordedRun:
    """One ``Orchestrator`` the CLI built -- recorded, never actually run."""

    goal: str
    settings: Settings
    raises: BaseException | None = None
    stop_calls: int = 0

    async def run(self) -> None:
        if self.raises is not None:
            raise self.raises

    def stop(self) -> None:
        self.stop_calls += 1


def _record_orchestrators(
    monkeypatch: pytest.MonkeyPatch, *, raises: BaseException | None = None
) -> list[_RecordedRun]:
    """Replace ``furrow.cli.main.Orchestrator`` with a recording fake.

    Returns the list the fake appends to, so a test can assert both what the CLI
    passed in and -- just as importantly -- that it built no orchestrator at all.
    """
    built: list[_RecordedRun] = []

    def factory(*, goal: str, settings: Settings) -> _RecordedRun:
        run = _RecordedRun(goal=goal, settings=settings, raises=raises)
        built.append(run)
        return run

    monkeypatch.setattr(cli_module, "Orchestrator", factory)
    return built


# --------------------------------------------------------------------------- #
# CLI: --help
# --------------------------------------------------------------------------- #


def test_start_help_lists_every_documented_option() -> None:
    result = _run_cli("start", "--help")

    assert result.exit_code == 0
    assert "[GOAL]" in result.stdout
    for option in (
        "--provider",
        "--model",
        "--max-cycles",
        "--max-parallel",
        "--workspace",
        "--test-command",
        "--state-file",
        "-v, --debug",
    ):
        assert option in result.stdout


def test_web_help_lists_every_documented_option() -> None:
    result = _run_cli("web", "--help")

    assert result.exit_code == 0
    for option in ("--host", "--port", "-v, --debug"):
        assert option in result.stdout


# --------------------------------------------------------------------------- #
# CLI: option -> Settings mapping
# --------------------------------------------------------------------------- #


def test_start_options_are_applied_to_the_settings_the_orchestrator_receives(
    monkeypatch: pytest.MonkeyPatch, tmp_workspace: Path
) -> None:
    state_file = tmp_workspace / "nested" / "state.json"
    built = _record_orchestrators(monkeypatch)

    result = _run_cli(
        "start",
        "ship it",
        "--provider",
        "openai",
        "--model",
        "gpt-4o",
        "--max-cycles",
        "4",
        "--max-parallel",
        "7",
        "--workspace",
        str(tmp_workspace),
        "--test-command",
        "pytest -q",
        "--state-file",
        str(state_file),
    )

    assert result.exit_code == 0
    assert len(built) == 1
    run = built[0]
    assert run.goal == "ship it"
    assert run.settings.provider is Provider.OPENAI
    assert run.settings.model == "gpt-4o"
    assert run.settings.max_cycles == 4
    assert run.settings.max_parallel_tasks == 7
    assert run.settings.workspace == tmp_workspace
    assert run.settings.test_command == "pytest -q"
    assert run.settings.state_file == state_file


@pytest.mark.parametrize(
    ("args", "read", "expected"),
    [
        pytest.param(
            ["--provider", "ollama"], lambda s: s.provider, Provider.OLLAMA, id="provider"
        ),
        pytest.param(["--model", "custom-model"], lambda s: s.model, "custom-model", id="model"),
        pytest.param(["--max-cycles", "6"], lambda s: s.max_cycles, 6, id="max-cycles"),
        pytest.param(
            ["--max-parallel", "9"], lambda s: s.max_parallel_tasks, 9, id="max-parallel"
        ),
        pytest.param(
            ["--test-command", "make check"], lambda s: s.test_command, "make check", id="command"
        ),
    ],
)
def test_start_single_option_maps_onto_its_settings_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_workspace: Path,
    args: list[str],
    read: Callable[[Settings], object],
    expected: object,
) -> None:
    built = _record_orchestrators(monkeypatch)

    result = _run_cli("start", "do it", "--workspace", str(tmp_workspace), *args)

    assert result.exit_code == 0
    assert len(built) == 1
    assert read(built[0].settings) == expected


def test_start_provider_choice_is_case_insensitive(
    monkeypatch: pytest.MonkeyPatch, tmp_workspace: Path
) -> None:
    built = _record_orchestrators(monkeypatch)

    result = _run_cli(
        "start", "do it", "--provider", "OpenAI", "--workspace", str(tmp_workspace)
    )

    assert result.exit_code == 0
    assert built[0].settings.provider is Provider.OPENAI


def test_start_inherits_every_unset_option_from_the_configured_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_workspace: Path
) -> None:
    built = _record_orchestrators(monkeypatch)

    result = _run_cli("start", "do it", "--workspace", str(tmp_workspace))

    assert result.exit_code == 0
    inherited = built[0].settings
    assert inherited.provider is settings.provider
    assert inherited.model == settings.model
    assert inherited.max_cycles == settings.max_cycles
    assert inherited.max_parallel_tasks == settings.max_parallel_tasks
    assert inherited.test_command == settings.test_command
    assert inherited.max_tokens == settings.max_tokens


def test_start_does_not_mutate_the_global_settings_singleton(
    monkeypatch: pytest.MonkeyPatch, tmp_workspace: Path
) -> None:
    before = settings.model_dump()
    _record_orchestrators(monkeypatch)

    result = _run_cli(
        "start",
        "ship it",
        "--provider",
        "openai",
        "--model",
        "gpt-4o",
        "--max-cycles",
        "4",
        "--max-parallel",
        "7",
        "--workspace",
        str(tmp_workspace),
        "--test-command",
        "pytest -q",
        "--state-file",
        str(tmp_workspace / "state.json"),
    )

    assert result.exit_code == 0
    assert settings.model_dump() == before


# --------------------------------------------------------------------------- #
# CLI: goal validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("goal", ["", " ", "   \t\n "])
def test_start_rejects_a_blank_goal_argument(
    monkeypatch: pytest.MonkeyPatch, goal: str
) -> None:
    built = _record_orchestrators(monkeypatch)

    result = _run_cli("start", goal)

    assert result.exit_code != 0
    assert "Error:" in result.stdout
    assert built == []


def test_start_rejects_a_blank_prompt_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    built = _record_orchestrators(monkeypatch)

    result = _run_cli("start", stdin="   \n")

    assert result.exit_code != 0
    assert "Error:" in result.stdout
    assert built == []


def test_start_aborts_when_the_goal_prompt_hits_end_of_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = _record_orchestrators(monkeypatch)

    result = _run_cli("start", stdin="")

    assert result.exit_code != 0
    assert "Aborted!" in result.stderr
    assert "Error:" not in result.stdout
    assert built == []


def test_start_never_dresses_an_aborted_prompt_up_as_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No goal plus end-of-input is a cancellation, not a runtime failure.

    Even under ``--debug`` it must stay a cancellation: the generic handler would
    otherwise print ``Error: run failed: Abort`` and a traceback for an ``Abort``
    that never had a failing frame behind it.
    """
    _record_orchestrators(monkeypatch)

    result = _run_cli("start", "--debug", stdin="")

    assert result.exit_code != 0
    assert "Aborted!" in result.stderr
    assert "Error:" not in result.stdout
    assert "Re-run with -v/--debug" not in result.output
    assert "Traceback (most recent call last)" not in result.output


# --------------------------------------------------------------------------- #
# CLI: failure reporting
# --------------------------------------------------------------------------- #


def test_start_prints_a_clean_error_for_a_failing_run(
    monkeypatch: pytest.MonkeyPatch, tmp_workspace: Path
) -> None:
    _record_orchestrators(monkeypatch, raises=RuntimeError("planner exploded"))

    result = _run_cli("start", "do it", "--workspace", str(tmp_workspace))

    assert result.exit_code == 1
    assert "Error:" in result.stdout
    assert "planner exploded" in result.stdout


def test_start_hides_the_traceback_until_debug_is_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_workspace: Path
) -> None:
    _record_orchestrators(monkeypatch, raises=RuntimeError("planner exploded"))

    quiet = _run_cli("start", "do it", "--workspace", str(tmp_workspace))
    verbose = _run_cli("start", "do it", "--workspace", str(tmp_workspace), "--debug")

    assert quiet.exit_code == 1
    assert "Traceback (most recent call last)" not in quiet.output
    assert verbose.exit_code == 1
    assert "Traceback (most recent call last)" in verbose.output


def test_start_surfaces_a_configuration_problem_with_an_credential_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_workspace: Path
) -> None:
    _record_orchestrators(
        monkeypatch, raises=ConfigError("ANTHROPIC_API_KEY is missing")
    )

    result = _run_cli("start", "do it", "--workspace", str(tmp_workspace))

    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY is missing" in result.stdout
    # The hint has to name the env var the user is expected to set.
    assert "ANTHROPIC_API_KEY=" in result.stdout
    assert "Traceback (most recent call last)" not in result.output


def test_start_rejects_a_workspace_that_is_not_an_existing_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built = _record_orchestrators(monkeypatch)
    missing = tmp_path / "not-created"

    result = _run_cli("start", "do it", "--workspace", str(missing))

    assert result.exit_code == 1
    assert "Error:" in result.stdout
    assert "--workspace" in result.stdout
    assert built == []


@pytest.mark.parametrize(
    ("option", "value"),
    [("--max-cycles", "-1"), ("--max-parallel", "0")],
    ids=["negative-cycles", "zero-parallelism"],
)
def test_start_rejects_an_out_of_range_loop_option(
    monkeypatch: pytest.MonkeyPatch, tmp_workspace: Path, option: str, value: str
) -> None:
    built = _record_orchestrators(monkeypatch)

    result = _run_cli("start", "do it", "--workspace", str(tmp_workspace), option, value)

    assert result.exit_code == 1
    assert "Error:" in result.stdout
    assert built == []


def test_start_treats_an_interrupt_as_a_clean_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_workspace: Path
) -> None:
    _record_orchestrators(monkeypatch, raises=KeyboardInterrupt())

    result = _run_cli("start", "do it", "--workspace", str(tmp_workspace))

    assert result.exit_code == 0
    assert "stopped by user" in result.stdout


# --------------------------------------------------------------------------- #
# CLI: furrow web
# --------------------------------------------------------------------------- #


def test_web_forwards_host_and_port_to_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(web, "run", lambda **kwargs: recorded.append(kwargs))

    result = _run_cli("web", "--host", "127.0.0.1", "--port", "8123")

    assert result.exit_code == 0
    assert recorded == [{"host": "127.0.0.1", "port": 8123}]


def test_web_serves_on_the_documented_default_interface(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(web, "run", lambda **kwargs: recorded.append(kwargs))

    result = _run_cli("web")

    assert result.exit_code == 0
    assert recorded == [{"host": "0.0.0.0", "port": 8000}]


def test_web_rejects_a_port_outside_the_valid_range(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(web, "run", lambda **kwargs: recorded.append(kwargs))

    result = _run_cli("web", "--port", "70000")

    # An out-of-range value is a usage error, exactly like a non-numeric one: click
    # owns both, so both exit 2 with the same click-formatted diagnostic.
    assert result.exit_code == 2
    assert "--port" in result.stderr
    assert "65535" in result.stderr
    assert recorded == []


@pytest.mark.parametrize(
    ("port", "accepted"),
    [("1", True), ("65535", True), ("0", False), ("65536", False)],
    ids=["lowest-valid", "highest-valid", "one-below", "one-above"],
)
def test_web_accepts_exactly_the_bindable_tcp_port_range(
    monkeypatch: pytest.MonkeyPatch, port: str, accepted: bool
) -> None:
    """Pin the ``--port`` bound to 1..65535 -- the ports a listener can bind.

    Asserting the two accepted boundaries is what makes the bound itself a
    pinned fact rather than an implementation detail; 8000 is the default and is
    covered by ``test_web_serves_on_the_documented_default_interface``.
    """
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(web, "run", lambda **kwargs: recorded.append(kwargs))

    result = _run_cli("web", "--port", port)

    if accepted:
        assert result.exit_code == 0
        assert recorded == [{"host": "0.0.0.0", "port": int(port)}]
    else:
        assert result.exit_code == 2
        assert "--port" in result.stderr
        assert recorded == []


def test_web_rejects_a_non_numeric_port_with_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(web, "run", lambda **kwargs: recorded.append(kwargs))

    result = _run_cli("web", "--port", "not-a-port")

    assert result.exit_code == 2
    assert "--port" in result.stderr
    assert recorded == []


def test_start_rejects_an_unknown_provider_with_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = _record_orchestrators(monkeypatch)

    result = _run_cli("start", "do it", "--provider", "nope")

    assert result.exit_code == 2
    assert "--provider" in result.stderr
    assert built == []


# --------------------------------------------------------------------------- #
# Web: the stubbed orchestrator and its event script
# --------------------------------------------------------------------------- #

#: One of every event ``Orchestrator`` emits, in the order it emits them. The
#: ``error`` entry is deliberately in the middle: an orchestrator ``error`` event
#: is forwarded like any other, and must stay distinct from the server's own
#: ``{"event": "error", "code": ...}`` control message.
_EVENT_SCRIPT: tuple[tuple[str, dict[str, Any]], ...] = (
    ("cycle_start", {"cycle": 1}),
    (
        "plan",
        {
            "rationale": "start small",
            "task_count": 1,
            "tasks": [
                {
                    "id": "t1",
                    "description": "add a button",
                    "files": ["app.py"],
                    "dependencies": [],
                    "status": "pending",
                }
            ],
        },
    ),
    ("task_start", {"id": "t1", "description": "add a button"}),
    (
        "task_done",
        {"id": "t1", "status": "completed", "result": "added button", "error": None},
    ),
    ("test_result", {"passed": True, "summary": "1 passed", "failures": []}),
    ("cycle_end", {"cycle": 1, "passed": True, "completed": ["t1"], "failed": []}),
    ("error", {"message": "planner hiccup", "where": "planner"}),
    ("done", {"reason": "All tasks completed and tests passed", "cycles": 1}),
)

#: The payload keys each orchestrator event carries, independent of the script
#: above: a streamed message must be the event name plus exactly these keys.
_EVENT_PAYLOAD_KEYS: dict[str, frozenset[str]] = {
    "cycle_start": frozenset({"cycle"}),
    "plan": frozenset({"rationale", "task_count", "tasks"}),
    "task_start": frozenset({"id", "description"}),
    "task_done": frozenset({"id", "status", "result", "error"}),
    "test_result": frozenset({"passed", "summary", "failures"}),
    "cycle_end": frozenset({"cycle", "passed", "completed", "failed"}),
    "done": frozenset({"reason", "cycles"}),
    "error": frozenset({"message", "where"}),
}

#: The shortest useful stream: one event, then the run parks until stopped.
_START_ONLY_EVENTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("cycle_start", {"cycle": 1}),
)


@dataclass
class _WebStubConfig:
    """Knobs a test turns to shape what the stubbed run does."""

    events: tuple[tuple[str, dict[str, Any]], ...] = _EVENT_SCRIPT
    block_until_stopped: bool = False
    raises: BaseException | None = None


class _WebStubOrchestrator:
    """Stand-in for :class:`Orchestrator` that replays a scripted event stream.

    With ``block_until_stopped`` the run parks on an :class:`asyncio.Event` after
    emitting its script, so the test can hold the single run slot open while it
    inspects the connection. The event is created here -- inside the app's event
    loop -- and set by :meth:`stop`, which the web layer calls on the same loop,
    so a stop can never be lost to a race.
    """

    def __init__(
        self,
        *,
        goal: str,
        on_event: Callable[[str, dict[str, Any]], None],
        settings: Settings,
        config: _WebStubConfig,
    ) -> None:
        self.goal = goal
        self.settings = settings
        self.stop_calls = 0
        self.finished = False
        self._emit_event = on_event
        self._config = config
        self._stopped = asyncio.Event()

    def stop(self) -> None:
        self.stop_calls += 1
        self._stopped.set()

    async def run(self) -> None:
        for name, payload in self._config.events:
            self._emit_event(name, payload)
        if self._config.block_until_stopped:
            await self._stopped.wait()
            self._emit_event("done", {"reason": "Stop requested", "cycles": 1})
        if self._config.raises is not None:
            raise self._config.raises
        self.finished = True


def _stub_web_orchestrator(
    monkeypatch: pytest.MonkeyPatch, config: _WebStubConfig | None = None
) -> tuple[list[_WebStubOrchestrator], _WebStubConfig]:
    """Replace ``furrow.web.server.Orchestrator`` with an event-scripting stub.

    Returns the list of stub instances -- one per construction, which is how a
    test proves a refused start really built nothing -- and the config, which a
    test may adjust between connections (e.g. to let a second run finish on its
    own).
    """
    resolved = config or _WebStubConfig()
    built: list[_WebStubOrchestrator] = []

    def factory(
        *,
        goal: str,
        on_event: Callable[[str, dict[str, Any]], None],
        settings: Settings,
    ) -> _WebStubOrchestrator:
        stub = _WebStubOrchestrator(
            goal=goal, on_event=on_event, settings=settings, config=resolved
        )
        built.append(stub)
        return stub

    monkeypatch.setattr(web, "Orchestrator", factory)
    return built, resolved


def _clear_run_registry() -> None:
    """Empty the web layer's active-run slot, under its own lock."""
    with web._registry_lock:
        web._active_run = None


def _wait_for_idle(timeout: float = 5.0) -> None:
    """Block until the web layer has released its active-run slot."""
    deadline = time.monotonic() + timeout
    while web.is_running():
        if time.monotonic() >= deadline:
            pytest.fail("the web layer never released its active-run slot")
        time.sleep(0.005)


def _receive_json(ws: WebSocketTestSession, timeout: float = 10.0) -> dict[str, Any]:
    """Receive one websocket message, failing instead of blocking forever.

    ``WebSocketTestSession.receive_json`` waits on the ASGI portal with no
    timeout at all, so a server that accepts the socket and then never forwards
    anything -- exactly the bug this file exists to catch -- would hang the whole
    suite instead of failing one test. The wait is therefore bounded by a worker
    thread; if it does expire the thread is left to die with the portal when the
    ``websocket_connect`` block unwinds.
    """
    pool = futures.ThreadPoolExecutor(max_workers=1)
    pending = pool.submit(ws.receive_json)
    try:
        message = pending.result(timeout=timeout)
    except futures.TimeoutError as exc:
        pool.shutdown(wait=False, cancel_futures=True)
        raise AssertionError(f"no websocket message within {timeout}s") from exc
    pool.shutdown(wait=True)
    return message


def _drain_until_run_finished(ws: WebSocketTestSession) -> list[dict[str, Any]]:
    """Read messages until the terminal ``run_finished`` envelope arrives."""
    received: list[dict[str, Any]] = []
    while True:
        message = _receive_json(ws)
        received.append(message)
        if message["event"] == "run_finished":
            return received


def _close_code(ws: WebSocketTestSession) -> int:
    """The code the server closed the socket with."""
    with pytest.raises(WebSocketDisconnect) as raised:
        _receive_json(ws)
    return int(raised.value.code)


# --------------------------------------------------------------------------- #
# Web: streaming
# --------------------------------------------------------------------------- #


def test_ws_start_streams_one_message_per_orchestrator_event(
    web_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_web_orchestrator(monkeypatch)

    with web_client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "add a button"})
        received = _drain_until_run_finished(ws)

    streamed = received[:-1]
    assert [message["event"] for message in streamed] == [name for name, _ in _EVENT_SCRIPT]
    for message, (name, payload) in zip(streamed, _EVENT_SCRIPT, strict=True):
        assert frozenset(message) == {"event"} | _EVENT_PAYLOAD_KEYS[name]
        assert message == {"event": name, **payload}


def test_ws_run_finishes_with_a_terminal_envelope_and_a_normal_close(
    web_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_web_orchestrator(monkeypatch)

    with web_client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "add a button"})
        received = _drain_until_run_finished(ws)

        assert received[-1] == {
            "event": "run_finished",
            "status": "completed",
            "goal": "add a button",
            "detail": "",
        }
        assert _close_code(ws) == 1000

    assert not web.is_running()


def test_ws_reports_a_failing_run_as_failed_and_frees_the_slot(
    web_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_web_orchestrator(
        monkeypatch, _WebStubConfig(events=(), raises=RuntimeError("planner crashed"))
    )

    with web_client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "explode"})
        received = _drain_until_run_finished(ws)

        assert received[-1] == {
            "event": "run_finished",
            "status": "failed",
            "goal": "explode",
            "detail": "planner crashed",
        }
        assert _close_code(ws) == 1000

    assert not web.is_running()


def test_ws_model_option_reaches_the_settings_of_the_run(
    web_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clean_env: None,
) -> None:
    # chdir keeps the layer's own ``Settings()`` away from any ambient ``.env``.
    monkeypatch.chdir(tmp_path)
    built, _ = _stub_web_orchestrator(monkeypatch)

    with web_client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "use another model", "model": "custom-model-id"})
        _drain_until_run_finished(ws)

    assert built[0].settings.model == "custom-model-id"


# --------------------------------------------------------------------------- #
# Web: control messages
# --------------------------------------------------------------------------- #


def test_ws_stop_action_stops_the_active_run(
    web_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    built, _ = _stub_web_orchestrator(
        monkeypatch, _WebStubConfig(events=_START_ONLY_EVENTS, block_until_stopped=True)
    )

    with web_client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "long run"})
        assert _receive_json(ws)["event"] == "cycle_start"
        assert web.is_running() is True

        ws.send_json({"action": "stop"})
        assert _receive_json(ws) == {"event": "stopping", "goal": "long run"}
        received = _drain_until_run_finished(ws)

    assert built[0].stop_calls == 1
    assert received[-1]["status"] == "stopped"
    assert not web.is_running()


def test_ws_non_json_message_is_answered_and_the_socket_stays_open(
    web_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    built, _ = _stub_web_orchestrator(
        monkeypatch, _WebStubConfig(events=_START_ONLY_EVENTS, block_until_stopped=True)
    )

    with web_client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "long run"})
        assert _receive_json(ws)["event"] == "cycle_start"

        ws.send_text("this is not json")
        reply = _receive_json(ws)
        assert reply["event"] == "error"
        assert reply["code"] == "bad_message"
        assert "this is not json" in reply["message"]

        # Still open: the same socket can still drive the run to its end.
        ws.send_json({"action": "stop"})
        assert _receive_json(ws)["event"] == "stopping"
        received = _drain_until_run_finished(ws)

    assert received[-1]["status"] == "stopped"
    assert built[0].stop_calls == 1


def test_ws_unsupported_action_is_answered_and_the_socket_stays_open(
    web_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_web_orchestrator(
        monkeypatch, _WebStubConfig(events=_START_ONLY_EVENTS, block_until_stopped=True)
    )

    with web_client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "long run"})
        assert _receive_json(ws)["event"] == "cycle_start"

        ws.send_json({"action": "pause"})
        reply = _receive_json(ws)
        assert reply["event"] == "error"
        assert reply["code"] == "bad_message"

        ws.send_json({"action": "stop"})
        assert _receive_json(ws)["event"] == "stopping"
        _drain_until_run_finished(ws)


# --------------------------------------------------------------------------- #
# Web: refusing to start
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload",
    [{"goal": ""}, {"goal": "   "}, {"model": "no goal here"}],
    ids=["empty-goal", "blank-goal", "missing-goal"],
)
def test_ws_invalid_start_payload_is_rejected_without_starting_a_run(
    web_client: TestClient, monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> None:
    built, _ = _stub_web_orchestrator(monkeypatch)

    with web_client.websocket_connect("/ws") as ws:
        ws.send_json(payload)
        rejection = _receive_json(ws)
        assert rejection["event"] == "error"
        assert rejection["code"] == "invalid_payload"
        assert _close_code(ws) == 1003

    assert built == []
    assert not web.is_running()


@pytest.mark.parametrize(
    "frame", ["not json at all", "[1, 2, 3]"], ids=["non-json", "not-an-object"]
)
def test_ws_unusable_first_frame_is_rejected_without_starting_a_run(
    web_client: TestClient, monkeypatch: pytest.MonkeyPatch, frame: str
) -> None:
    built, _ = _stub_web_orchestrator(monkeypatch)

    with web_client.websocket_connect("/ws") as ws:
        ws.send_text(frame)
        rejection = _receive_json(ws)
        assert rejection["event"] == "error"
        assert rejection["code"] == "bad_message"
        assert _close_code(ws) == 1003

    assert built == []
    assert not web.is_running()


def test_ws_second_concurrent_start_is_refused_without_building_an_orchestrator(
    web_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    built, _ = _stub_web_orchestrator(
        monkeypatch, _WebStubConfig(events=_START_ONLY_EVENTS, block_until_stopped=True)
    )

    with web_client.websocket_connect("/ws") as first:
        first.send_json({"goal": "first run"})
        assert _receive_json(first)["event"] == "cycle_start"

        with web_client.websocket_connect("/ws") as second:
            second.send_json({"goal": "second run"})
            rejection = _receive_json(second)
            assert rejection["event"] == "error"
            assert rejection["code"] == "run_in_progress"
            assert _close_code(second) == 4001

        # The refused start must not have competed for the workspace.
        assert len(built) == 1
        assert built[0].goal == "first run"

        first.send_json({"action": "stop"})
        _drain_until_run_finished(first)

    assert not web.is_running()


# --------------------------------------------------------------------------- #
# Web: the single-run slot survives a vanishing client
# --------------------------------------------------------------------------- #


def test_ws_client_disconnect_mid_run_frees_the_slot_for_the_next_client(
    web_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _WebStubConfig(events=_START_ONLY_EVENTS, block_until_stopped=True)
    built, config = _stub_web_orchestrator(monkeypatch, config)

    with web_client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "client walks away"})
        assert _receive_json(ws)["event"] == "cycle_start"
        assert web.is_running() is True

    # Leaving the context disconnects the client while the run is still going.
    _wait_for_idle()
    assert not web.is_running()
    assert built[0].stop_calls == 1

    # The server must not be wedged: a new client gets a working run.
    config.block_until_stopped = False
    with web_client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "second client"})
        received = _drain_until_run_finished(ws)

    assert received[-1]["status"] == "completed"
    assert len(built) == 2
    assert built[1].goal == "second client"
    assert not web.is_running()


def test_is_running_is_false_while_no_websocket_run_is_active() -> None:
    assert web.is_running() is False


# --------------------------------------------------------------------------- #
# Web: the HTML page
# --------------------------------------------------------------------------- #


def test_index_serves_the_goal_form_and_its_websocket_wiring(
    web_client: TestClient,
) -> None:
    response = web_client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert 'id="goal"' in body
    assert "/ws" in body
    assert "new WebSocket(" in body
    # The regression this page exists for: it opened a socket and never sent a
    # goal, so the browser sat on "Starting..." forever.
    assert "socket.send(JSON.stringify({ goal: goal }))" in body
