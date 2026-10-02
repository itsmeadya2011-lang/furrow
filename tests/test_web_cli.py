"""Tests for the websocket streaming endpoint and the CLI option plumbing.

Both surfaces are exercised in-process: the orchestrator is replaced by a stub
that emits scripted events, so nothing here can reach a live model.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, ClassVar

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

import furrow.cli.main as cli_main
import furrow.config as config_module
from furrow.config import Provider, Settings
from furrow.web import server

GOAL = "build the thing"


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #


class StubOrchestrator:
    """Emits two events through ``on_event`` and returns.

    ``on_event`` is deliberately a *synchronous* attribute read, exactly like
    the real orchestrator's, so this fails loudly if the endpoint ever tries to
    await the hook.
    """

    def __init__(self, goal: str, client: Any = None, settings: Any = None) -> None:
        self.goal = goal
        self.client = client
        self.settings = settings
        self.on_event: Callable[[dict[str, Any]], None] | None = None
        self.cycles = 2
        self.seen: list[dict[str, Any]] = []

    async def run(self) -> None:
        hook = self.on_event
        for event in (
            {"type": "cycle_start", "goal": self.goal, "cycle": 1},
            {"type": "halted", "reason": "scripted halt", "cycles": 2, "cycle": 2},
        ):
            self.seen.append(event)
            if hook is not None:
                hook(event)


class ExplodingOrchestrator(StubOrchestrator):
    async def run(self) -> None:
        raise RuntimeError("the orchestrator fell over")


class RecordingOrchestrator:
    """CLI stub that only remembers what it was constructed with."""

    received: ClassVar[list[Any]] = []
    goals: ClassVar[list[str]] = []
    error: BaseException | None = None

    def __init__(self, goal: str, client: Any = None, settings: Any = None) -> None:
        self.goal = goal
        self.settings = settings
        type(self).received.append(settings)
        type(self).goals.append(goal)

    async def run(self) -> None:
        if type(self).error is not None:
            raise type(self).error


@pytest.fixture(autouse=True)
def reset_recording_orchestrator() -> None:
    RecordingOrchestrator.received = []
    RecordingOrchestrator.goals = []
    RecordingOrchestrator.error = None


class SentRecorder:
    """Wraps the real ASGI app and remembers every message the server sends.

    ``TestClient`` offers no non-blocking "is anything else queued?" probe: an
    endpoint that simply returns never queues a ``websocket.close``, so a
    trailing ``receive_text()`` waits on a message that will never arrive and
    the test deadlocks. Recording the raw ASGI ``send`` calls is what lets a
    test assert "the server stopped after the terminal frame" without ever
    reading past the frames it knows about.
    """

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.finished = False

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        async def recording_send(message: dict[str, Any]) -> None:
            self.sent.append(message)
            await send(message)

        try:
            await server.app(scope, receive, recording_send)
        finally:
            self.finished = True

    def frame_types(self) -> list[str]:
        """Every text frame the endpoint put on the wire, in order."""
        return [
            json.loads(message["text"])["type"]
            for message in self.sent
            if message["type"] == "websocket.send"
        ]

    def await_finished(self, timeout: float = 2.0) -> None:
        """Block - bounded by ``timeout`` - until the endpoint has returned.

        The endpoint finishes on its own and never waits on the client, so this
        normally returns immediately; it is here so the recorded frame list is
        authoritative once the assertions run. A still-running endpoint fails
        the test instead of silently hiding a trailing frame.
        """
        deadline = time.monotonic() + timeout
        while not self.finished and time.monotonic() < deadline:
            time.sleep(0.005)
        assert self.finished, "the websocket endpoint never returned"


# --------------------------------------------------------------------------- #
# 1. websocket streaming
# --------------------------------------------------------------------------- #


def test_websocket_streams_every_event_then_one_terminal_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "Orchestrator", StubOrchestrator)

    with TestClient(server.app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": GOAL})
        frames = [json.loads(ws.receive_text()) for _ in range(3)]

    assert [frame["type"] for frame in frames] == ["cycle_start", "halted", "done"]
    assert frames[0]["goal"] == GOAL
    assert frames[1]["reason"] == "scripted halt"
    assert frames[2] == {"type": "done", "cycles": 2}


def test_websocket_ends_with_exactly_one_terminal_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """The connection must stop right after ``done`` - no trailing duplicate."""
    monkeypatch.setattr(server, "Orchestrator", StubOrchestrator)
    recorder = SentRecorder()

    with TestClient(recorder) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": GOAL})
        # Exactly the three frames the stub run is scripted to produce. The
        # read count is the bound: a fourth read would block forever, so the
        # "nothing after ``done``" claim is made against the recorded
        # sends instead (see ``SentRecorder``).
        types = [json.loads(ws.receive_text())["type"] for _ in range(3)]
        recorder.await_finished()

    wire = recorder.frame_types()
    assert types == wire, "the client did not see every frame the server sent"
    assert wire == ["cycle_start", "halted", "done"]
    assert wire.count("done") == 1
    assert wire[-1] == "done"


def test_websocket_passes_the_requested_model_to_the_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[StubOrchestrator] = []

    def factory(goal: str, client: Any = None, settings: Any = None) -> StubOrchestrator:
        orchestrator = StubOrchestrator(goal, client=client, settings=settings)
        captured.append(orchestrator)
        return orchestrator

    monkeypatch.setattr(server, "Orchestrator", factory)

    with TestClient(server.app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": GOAL, "model": "gpt-4o-mini"})
        json.loads(ws.receive_text())
        json.loads(ws.receive_text())
        json.loads(ws.receive_text())

    assert captured[0].settings is not None
    assert captured[0].settings.model == "gpt-4o-mini"
    # The per-connection copy must not have touched the process-wide singleton.
    assert config_module.settings.model != "gpt-4o-mini"


def test_websocket_without_a_model_never_forges_a_settings_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "Orchestrator", StubOrchestrator)

    with TestClient(server.app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": GOAL})
        for _ in range(3):
            json.loads(ws.receive_text())

    assert server.settings_for_model(None) is None
    assert server.settings_for_model("") is None


def test_websocket_reports_an_orchestrator_failure_as_an_error_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "Orchestrator", ExplodingOrchestrator)

    with TestClient(server.app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": GOAL})
        frame = json.loads(ws.receive_text())

    assert frame == {"type": "error", "message": "the orchestrator fell over"}


def test_websocket_rejects_an_empty_goal_without_running_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args: Any, **kwargs: Any) -> StubOrchestrator:
        raise AssertionError("an empty goal must not start an orchestrator")

    monkeypatch.setattr(server, "Orchestrator", explode)

    with TestClient(server.app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "   "})
        frame = json.loads(ws.receive_text())

    assert frame["type"] == "error"
    assert "goal" in frame["message"]


def test_websocket_survives_a_client_that_hangs_up_mid_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disconnect must not raise out of the endpoint or wedge the writer."""
    monkeypatch.setattr(server, "Orchestrator", ExplodingOrchestrator)

    with TestClient(server.app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": GOAL})

    # Reaching here without the server blowing up is the assertion.


# --------------------------------------------------------------------------- #
# 2. event framing
# --------------------------------------------------------------------------- #


def test_event_frames_are_bounded() -> None:
    frame = server._encode_event(
        {"type": "task_completed", "cycle": 1, "result": "x" * 50_000}
    )

    assert len(frame.encode("utf-8")) <= server.MAX_EVENT_BYTES
    payload = json.loads(frame)
    assert payload["type"] == "task_completed"
    assert payload["cycle"] == 1
    assert payload["result"].endswith("[truncated]")
    assert len(payload["result"]) < 50_000


def test_a_wide_event_still_fits_inside_the_byte_cap() -> None:
    frame = server._encode_event(
        {"type": "plan", "cycle": 3, **{f"task_ids_{i}": "t" * 900 for i in range(40)}}
    )

    assert len(frame.encode("utf-8")) <= server.MAX_EVENT_BYTES
    payload = json.loads(frame)  # still valid JSON after the hard slice
    assert payload["type"] == "plan"
    assert payload["truncated"] is True


def test_ordinary_events_round_trip_untouched() -> None:
    event = {"type": "test_result", "cycle": 2, "passed": True, "summary": "green"}

    assert json.loads(server._encode_event(event)) == event


def test_settings_for_model_copies_rather_than_mutates() -> None:
    before = config_module.settings.model_dump()

    copy = server.settings_for_model("some-other-model")

    assert copy is not None
    assert copy.model == "some-other-model"
    assert config_module.settings.model_dump() == before


# --------------------------------------------------------------------------- #
# 3. CLI options
# --------------------------------------------------------------------------- #


def test_cli_start_forwards_every_option_onto_a_settings_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_main, "Orchestrator", RecordingOrchestrator)
    before = config_module.settings.model_dump()

    result = CliRunner().invoke(
        cli_main.main,
        [
            "start",
            GOAL,
            "--model",
            "gpt-4o",
            "--max-cycles",
            "7",
            "--max-parallel-tasks",
            "3",
            "--provider",
            "openai",
        ],
    )

    assert result.exit_code == 0, result.output
    assert RecordingOrchestrator.goals == [GOAL]
    run_settings = RecordingOrchestrator.received[0]
    assert isinstance(run_settings, Settings)
    assert run_settings is not config_module.settings
    assert run_settings.model == "gpt-4o"
    assert run_settings.max_cycles == 7
    assert run_settings.max_parallel_tasks == 3
    assert run_settings.provider is Provider.OPENAI
    # The whole point: the process-wide singleton is untouched.
    assert config_module.settings.model_dump() == before


def test_cli_start_leaves_omitted_options_at_their_configured_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing option must not overwrite a value that came from the env."""
    monkeypatch.setattr(cli_main, "Orchestrator", RecordingOrchestrator)
    configured = Settings(
        model="env-model", max_cycles=5, max_parallel_tasks=2, provider=Provider.OLLAMA
    )
    monkeypatch.setattr(cli_main, "settings", configured)

    result = CliRunner().invoke(cli_main.main, ["start", GOAL])

    assert result.exit_code == 0, result.output
    run_settings = RecordingOrchestrator.received[0]
    assert run_settings.model == "env-model"
    assert run_settings.max_cycles == 5
    assert run_settings.max_parallel_tasks == 2
    assert run_settings.provider is Provider.OLLAMA


def test_cli_treats_an_explicit_zero_as_a_real_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--max-cycles 0`` means "unlimited" and must not read as "unset"."""
    monkeypatch.setattr(cli_main, "Orchestrator", RecordingOrchestrator)
    monkeypatch.setattr(cli_main, "settings", Settings(max_cycles=5, max_parallel_tasks=2))

    result = CliRunner().invoke(cli_main.main, ["start", GOAL, "--max-cycles", "0"])

    assert result.exit_code == 0, result.output
    run_settings = RecordingOrchestrator.received[0]
    assert run_settings.max_cycles == 0
    # ... while an omitted option still keeps the configured value.
    assert run_settings.max_parallel_tasks == 2


def test_cli_rejects_an_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_main, "Orchestrator", RecordingOrchestrator)

    result = CliRunner().invoke(cli_main.main, ["start", GOAL, "--provider", "nope"])

    assert result.exit_code != 0
    assert RecordingOrchestrator.received == []


def test_cli_prompts_for_a_goal_when_none_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_main, "Orchestrator", RecordingOrchestrator)

    result = CliRunner().invoke(cli_main.main, ["start"], input=f"{GOAL}\n")

    assert result.exit_code == 0, result.output
    assert RecordingOrchestrator.goals == [GOAL]


def test_cli_swallows_a_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_main, "Orchestrator", RecordingOrchestrator)
    RecordingOrchestrator.error = KeyboardInterrupt()

    result = CliRunner().invoke(cli_main.main, ["start", GOAL])

    assert result.exit_code == 0, result.output
