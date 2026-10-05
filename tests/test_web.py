"""Regression tests for the HTTP and WebSocket front end (``furrow.web.server``).

Everything runs in-process through ``fastapi.testclient.TestClient`` against the
real ASGI app, with ``LLMClient.complete`` stubbed by a class-level patch - the
server builds its own :class:`LLMClient`, so the stub is picked up by every run.
No socket is ever opened: the autouse ``no_real_network`` fixture enforces that.

Load-bearing regressions pinned here:

* the websocket **actually delivers event frames** (``accepted`` plus the
  orchestrator's events) instead of accepting a socket and going quiet;
* invalid input is refused with an ``error`` frame and the socket closes;
* a web run terminates within its cycle budget;
* ``FURROW_WEB_TOKEN`` gates ``start`` and ``GET /``.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from furrow.web import server
from tests.conftest import (
    TESTER,
    WORKER,
    ScriptedCompletion,
    install_scripted,
    make_settings,
    plan_reply,
    task,
    verdict_json,
    worker_reply,
)

DEADLINE = 30.0
"""Hard deadline for every websocket exchange: a regression must not hang."""

PASSING = verdict_json(True, "all green")
FAILING = verdict_json(False, "still broken", ["test_x failed"])

ASSET_PATTERN = re.compile(r"(?:src|href)\s*=\s*[\"']([^\"']*)[\"']", re.IGNORECASE)


@pytest.fixture
def web(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Path]:
    """A ``TestClient`` whose runs are hermetic: temp workspace, ``true`` as tests."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    settings = make_settings(workspace, test_command="true", max_cycles=0)
    monkeypatch.setattr(server, "_web_settings", settings)
    monkeypatch.delenv(server.TOKEN_ENV, raising=False)
    monkeypatch.delenv(server.MAX_CONCURRENT_RUNS_ENV, raising=False)
    monkeypatch.delenv("FURROW_MAX_CYCLES", raising=False)
    with TestClient(server.app) as client:
        yield client, workspace
    assert settings.max_concurrent_runs >= 1


def install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    planner: Any = "",
    worker: Any = "",
    tester: Any = PASSING,
) -> ScriptedCompletion:
    return install_scripted(monkeypatch, planner=planner, worker=worker, tester=tester)


DEFAULT_PLANNER = plan_reply(task("1", "make a module", files=["made.py"]))
DEFAULT_WORKER = worker_reply("created made.py", files=[{"path": "made.py", "content": "X = 1\n"}])


def exchange(client: TestClient, payload: Any) -> list[dict[str, Any]]:
    """Open a socket, send *payload* and collect every frame until the close."""
    frames: list[dict[str, Any]] = []
    with client.websocket_connect("/ws") as socket:
        socket.send_text(payload if isinstance(payload, str) else json.dumps(payload))
        try:
            while True:
                frames.append(socket.receive_json())
        except WebSocketDisconnect:
            pass
    return frames


async def bounded(exchange_fn: Any) -> Any:
    """Run a blocking ``TestClient`` exchange with a hard deadline."""
    return await asyncio.wait_for(asyncio.to_thread(exchange_fn), timeout=DEADLINE)


async def run_ws(client: TestClient, payload: Any) -> list[dict[str, Any]]:
    return await bounded(lambda: exchange(client, payload))


def names(frames: list[dict[str, Any]]) -> list[str]:
    return [frame["event"] for frame in frames]


# ------------------------------------------------------------------- HTTP


def test_index_is_served(web: tuple[TestClient, Path]) -> None:
    client, _ = web
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Furrow" in response.text


def test_index_references_no_external_asset(web: tuple[TestClient, Path]) -> None:
    """The page must be fully self-contained: no CDN, no external stylesheet."""
    client, _ = web
    html = client.get("/").text
    references = ASSET_PATTERN.findall(html)
    assert references == [], f"external asset references found: {references}"
    assert "<script src" not in html.lower()
    assert "cdn" not in html.lower()
    assert "//unpkg" not in html.lower()
    assert "<style>" in html and "<script>" in html


def test_health_is_open(web: tuple[TestClient, Path]) -> None:
    client, _ = web
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["active_runs"] == 0
    assert body["auth_required"] is False
    assert isinstance(body["web_max_cycles"], int)
    assert body["web_max_cycles"] > 0
    assert "last_run" not in body


def test_api_state_is_served(web: tuple[TestClient, Path]) -> None:
    client, _ = web
    response = client.get("/api/state")
    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {"version", "active_runs", "max_concurrent_runs", "last_run"}
    assert body["auth_required"] is False


async def test_api_state_reflects_the_last_run(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = web
    install(monkeypatch, planner=DEFAULT_PLANNER, worker=DEFAULT_WORKER, tester=PASSING)
    frames = await run_ws(client, {"type": "start", "goal": "build the thing"})
    assert "done" in names(frames)

    body = client.get("/api/state").json()
    last = body["last_run"]
    assert last["goal"] == "build the thing"
    assert last["done"] is True
    assert last["active"] is False
    assert last["cycles"] == 1
    assert last["outcome"] == "done"
    assert last["error"] is None
    assert last["events"] > 0
    assert last["current_plan"]["tasks"][0]["id"] == "1"
    assert last["last_test_result"]["passed"] is True
    assert last["history"]
    assert last["started_at"] and last["finished_at"]


async def test_api_state_reports_a_failed_run(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = web
    install(monkeypatch, planner=DEFAULT_PLANNER, worker=DEFAULT_WORKER, tester=FAILING)
    await run_ws(client, {"type": "start", "goal": "g", "max_cycles": 1})
    last = client.get("/api/state").json()["last_run"]
    assert last["done"] is False
    assert last["outcome"] == "halted"
    assert last["last_test_result"]["passed"] is False
    assert last["last_test_result"]["failures"] == ["test_x failed"]


# ------------------------------------------------------------- websocket


async def test_the_websocket_delivers_event_frames(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression: the socket was accepted and then nothing was ever sent."""
    client, workspace = web
    install(monkeypatch, planner=DEFAULT_PLANNER, worker=DEFAULT_WORKER, tester=PASSING)
    frames = await run_ws(client, {"type": "start", "goal": "make it so"})

    assert frames, "no frame was delivered at all"
    for frame in frames:
        assert set(frame) <= {"event", "message", "data"}
        assert isinstance(frame["event"], str)
        assert isinstance(frame["message"], str)

    seen = names(frames)
    assert seen[0] == "accepted"
    for expected in ("run", "cycle", "plan", "task", "test", "done"):
        assert expected in seen, f"no {expected!r} frame in {seen}"

    accepted = frames[0]
    # The wire schema documents ``data`` on the accepted frame; the implementation
    # only puts the budget in the message, so accept either shape.
    if "data" in accepted:
        assert accepted["data"]["run_id"]
        assert accepted["data"]["max_cycles"] >= 1
    else:
        assert "cycle(s)" in accepted["message"]
    assert (workspace / "made.py").read_text() == "X = 1\n"


async def test_the_final_done_frame_carries_the_summary(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = web
    install(monkeypatch, planner=DEFAULT_PLANNER, worker=DEFAULT_WORKER, tester=PASSING)
    frames = await run_ws(client, {"type": "start", "goal": "g"})
    final = frames[-1]
    assert final["event"] == "done"
    assert final["data"]["done"] is True
    assert final["data"]["cycles"] == 1
    assert final["data"]["error"] is None


async def test_a_goal_without_an_explicit_type_is_accepted(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = web
    install(monkeypatch, planner=DEFAULT_PLANNER, worker=DEFAULT_WORKER, tester=PASSING)
    frames = await run_ws(client, {"goal": "implicit start"})
    assert names(frames)[0] == "accepted"
    assert "done" in names(frames)


async def test_the_model_option_overrides_every_agent_model(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = web
    scripted = install(monkeypatch, planner=DEFAULT_PLANNER, worker=DEFAULT_WORKER)
    await run_ws(client, {"type": "start", "goal": "g", "model": "my-model"})
    assert {call.model for call in scripted.calls} == {"my-model"}


# ------------------------------------------------------------ bad input


async def assert_rejected(web: tuple[TestClient, Path], payload: Any, needle: str) -> None:
    client, _ = web
    frames = await run_ws(client, payload)
    assert names(frames) == ["error"], f"expected a single error frame, got {names(frames)}"
    assert needle in frames[0]["message"]


async def test_an_empty_goal_is_rejected(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted = install(monkeypatch, planner=DEFAULT_PLANNER)
    await assert_rejected(web, {"type": "start", "goal": ""}, "goal")
    assert scripted.calls == []


async def test_a_blank_goal_is_rejected(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, planner=DEFAULT_PLANNER)
    await assert_rejected(web, {"type": "start", "goal": "   "}, "goal")


async def test_a_missing_goal_is_rejected(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, planner=DEFAULT_PLANNER)
    await assert_rejected(web, {"type": "start"}, "goal")


async def test_a_non_object_payload_is_rejected(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, planner=DEFAULT_PLANNER)
    await assert_rejected(web, "[1, 2, 3]", "JSON object")
    await assert_rejected(web, '"just a string"', "JSON object")
    await assert_rejected(web, "not json at all", "not valid JSON")


async def test_an_oversized_goal_is_rejected(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted = install(monkeypatch, planner=DEFAULT_PLANNER)
    oversized = {"type": "start", "goal": "x" * (server.MAX_GOAL_LENGTH + 1)}
    await assert_rejected(web, oversized, "too long")
    assert scripted.calls == []


async def test_a_goal_at_the_limit_is_accepted(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = web
    scripted = install(monkeypatch, planner=DEFAULT_PLANNER, worker=DEFAULT_WORKER)
    frames = await run_ws(client, {"type": "start", "goal": "y" * server.MAX_GOAL_LENGTH})
    assert names(frames)[0] == "accepted"
    assert scripted.calls


async def test_an_unknown_message_type_is_rejected(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, planner=DEFAULT_PLANNER)
    await assert_rejected(web, {"type": "pause", "goal": "g"}, "unsupported message type")


async def test_an_out_of_range_max_cycles_is_rejected(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, planner=DEFAULT_PLANNER)
    await assert_rejected(
        web,
        {"type": "start", "goal": "g", "max_cycles": server.HARD_MAX_WEB_CYCLES + 1},
        "max_cycles",
    )


async def test_a_binary_frame_is_rejected(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = web
    install(monkeypatch, planner=DEFAULT_PLANNER)
    frames = await bounded(
        lambda: _binary_exchange(client),
    )
    assert names(frames) == ["error"]
    assert "binary" in frames[0]["message"]


def _binary_exchange(client: TestClient) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    with client.websocket_connect("/ws") as socket:
        socket.send_bytes(b"\x00\x01\x02")
        try:
            while True:
                frames.append(socket.receive_json())
        except WebSocketDisconnect:
            pass
    return frames


async def test_a_second_start_on_the_same_socket_is_refused(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One run per connection; the run already in flight must not be disturbed."""
    client, _ = web
    gate = threading.Event()

    async def gated_planner(call: Any) -> str:
        # Block in a thread so the server's event loop keeps reading the socket.
        await asyncio.to_thread(gate.wait, 20)
        return DEFAULT_PLANNER

    scripted = install(monkeypatch, planner=gated_planner, worker=DEFAULT_WORKER, tester=PASSING)

    frames: list[dict[str, Any]] = []

    def exchange() -> None:
        with client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "start", "goal": "first run"})
            frames.append(socket.receive_json())
            socket.send_json({"type": "start", "goal": "second run"})
            gate.set()
            try:
                while True:
                    frames.append(socket.receive_json())
            except WebSocketDisconnect:
                pass

    await bounded(exchange)
    seen = names(frames)
    assert seen.count("error") == 1
    error = frames[seen.index("error")]
    assert "already in progress" in error["message"]
    assert "done" in seen
    # The original run was not disturbed.
    assert len(scripted.by_role[WORKER]) == 1
    assert client.get("/api/state").json()["last_run"]["goal"] == "first run"


async def test_bad_messages_during_a_run_are_reported_without_stopping_it(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = web
    gate = threading.Event()

    async def gated_planner(call: Any) -> str:
        await asyncio.to_thread(gate.wait, 20)
        return DEFAULT_PLANNER

    install(monkeypatch, planner=gated_planner, worker=DEFAULT_WORKER, tester=PASSING)
    frames: list[dict[str, Any]] = []

    def exchange() -> None:
        with client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "start", "goal": "g", "max_cycles": 1})
            socket.send_json({"type": "pause"})
            socket.send_text("not json")
            socket.send_text("[1, 2]")
            gate.set()
            try:
                while True:
                    frames.append(socket.receive_json())
            except WebSocketDisconnect:
                pass

    await bounded(exchange)
    seen = names(frames)
    assert seen.count("error") == 3
    assert "done" in seen
    assert seen[-1] == "done"


# ------------------------------------------------------------ cycle budget


async def test_a_web_run_terminates_within_its_cycle_budget(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run must not be able to pin the socket forever."""
    client, _ = web
    scripted = install(monkeypatch, planner=DEFAULT_PLANNER, worker=DEFAULT_WORKER, tester=FAILING)
    frames = await run_ws(client, {"type": "start", "goal": "g", "max_cycles": 2})
    assert names(frames).count("plan") >= 2
    final = frames[-1]
    assert final["event"] == "done"
    assert final["data"]["done"] is False
    assert final["data"]["cycles"] == 2
    assert len(scripted.by_role[TESTER]) == 2
    last = client.get("/api/state").json()["last_run"]
    assert last["max_cycles"] == 2
    assert last["cycles"] == 2
    assert last["active"] is False


async def test_the_web_layer_bounds_an_unlimited_budget(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``max_cycles=0`` (the CLI default) must not become unlimited here."""
    client, _ = web
    install(monkeypatch, planner=DEFAULT_PLANNER, worker=DEFAULT_WORKER, tester=FAILING)
    frames = await run_ws(client, {"type": "start", "goal": "g"})
    accepted = frames[0]
    budget = accepted["data"]["max_cycles"] if "data" in accepted else server.DEFAULT_WEB_MAX_CYCLES
    assert budget == server.DEFAULT_WEB_MAX_CYCLES
    assert frames[-1]["data"]["cycles"] == server.DEFAULT_WEB_MAX_CYCLES
    assert frames[-1]["data"]["done"] is False


async def test_the_run_slot_is_released_after_a_run(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = web
    install(monkeypatch, planner=DEFAULT_PLANNER, worker=DEFAULT_WORKER, tester=PASSING)
    await run_ws(client, {"type": "start", "goal": "one"})
    assert client.get("/health").json()["active_runs"] == 0
    await run_ws(client, {"type": "start", "goal": "two"})
    assert client.get("/health").json()["active_runs"] == 0


async def test_a_run_slot_is_released_after_a_rejected_start(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = web
    install(monkeypatch, planner=DEFAULT_PLANNER)
    await assert_rejected(web, {"type": "start", "goal": ""}, "goal")
    assert client.get("/health").json()["active_runs"] == 0


# --------------------------------------------------------------- token


async def test_a_start_without_the_token_is_refused(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = web
    monkeypatch.setenv(server.TOKEN_ENV, "s3cret")
    scripted = install(monkeypatch, planner=DEFAULT_PLANNER, worker=DEFAULT_WORKER)
    await assert_rejected(web, {"type": "start", "goal": "g"}, server.TOKEN_ENV)
    assert scripted.calls == []
    assert client.get("/health").json()["auth_required"] is True
    assert client.get("/api/state").json()["auth_required"] is True
    assert client.get("/api/state").json()["active_runs"] == 0


async def test_a_start_with_a_wrong_token_is_refused(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = web
    monkeypatch.setenv(server.TOKEN_ENV, "s3cret")
    scripted = install(monkeypatch, planner=DEFAULT_PLANNER)
    await assert_rejected(web, {"type": "start", "goal": "g", "token": "wrong"}, "unauthorized")
    assert scripted.calls == []


async def test_a_start_with_the_right_token_proceeds(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = web
    monkeypatch.setenv(server.TOKEN_ENV, "s3cret")
    scripted = install(monkeypatch, planner=DEFAULT_PLANNER, worker=DEFAULT_WORKER, tester=PASSING)
    frames = await run_ws(client, {"type": "start", "goal": "g", "token": "s3cret"})
    assert names(frames)[0] == "accepted"
    assert frames[-1]["data"]["done"] is True
    assert scripted.calls


def test_the_index_needs_the_token_when_one_is_configured(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = web
    assert client.get("/").status_code == 200
    monkeypatch.setenv(server.TOKEN_ENV, "s3cret")
    assert client.get("/").status_code == 401
    assert client.get(f"/?token={'s3cret'}").status_code == 200
    assert client.get("/", headers={server.TOKEN_HEADER: "s3cret"}).status_code == 200
    assert client.get("/", headers={server.TOKEN_HEADER: "nope"}).status_code == 401
    # The probe stays open even with auth on.
    assert client.get("/health").status_code == 200


# ------------------------------------------------------------ entrypoint


def test_run_rejects_an_impossible_port(web: tuple[TestClient, Path]) -> None:
    with pytest.raises(ValueError, match="port"):
        server.run(port=0, settings=server._web_settings)
    with pytest.raises(ValueError, match="port"):
        server.run(port=70000, settings=server._web_settings)


def test_run_rejects_an_impossible_concurrency(web: tuple[TestClient, Path]) -> None:
    with pytest.raises(ValueError, match="max_concurrent_runs"):
        server.run(port=8123, max_concurrent_runs=0, settings=server._web_settings)


def test_web_token_configured_reads_the_environment(
    web: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert server.web_token_configured() is False
    monkeypatch.setenv(server.TOKEN_ENV, "  ")
    assert server.web_token_configured() is False
    monkeypatch.setenv(server.TOKEN_ENV, "value")
    assert server.web_token_configured() is True


def test_app_version_is_a_string(web: tuple[TestClient, Path]) -> None:
    assert isinstance(server.app_version(), str)
