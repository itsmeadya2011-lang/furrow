"""End-to-end tests for :mod:`furrow.core` driving the real agents.

Everything below the orchestrator is real here: the real :class:`PlannerAgent` parses the model
reply, the real :class:`WorkerAgent` writes to a real temporary workspace, and the real
:class:`TesterAgent` interprets a subprocess exit code. Only the two edges are controlled — the
model replies come from the fake LLM client and the test command is a scripted subprocess — so
the whole run is hermetic.

The unit-level coverage lives in ``test_config.py``, ``test_llm.py``, ``test_agents.py`` and
``test_orchestrator.py``; this file exists to prove those pieces compose.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from furrow.agents import PLANNER_PROMPT, WORKER_PROMPT
from furrow.core.orchestrator import LoopResult, Orchestrator
from tests.conftest import EventRecorder, FakeLLMClient, ProcessOutcome, SubprocessStub

pytestmark = pytest.mark.usefixtures("captured_console")

RUN_TIMEOUT = 5.0


async def run_loop(
    client: FakeLLMClient,
    goal: str,
    on_event: Callable[[str, dict[str, Any]], Any] | None,
    bounded: Callable[..., Awaitable[Any]],
) -> tuple[LoopResult, Orchestrator]:
    """Run the real loop to completion under a deadline.

    ``Orchestrator.run`` turns a cancellation into a ``LoopResult``, so a hang would otherwise be
    indistinguishable from a legitimate stop; checking the reason keeps that explicit.
    """
    orchestrator = Orchestrator(goal=goal, client=client, on_event=on_event)
    result = await bounded(orchestrator.run(), RUN_TIMEOUT)
    assert result.stopped_reason != "cancelled", "the loop hit its deadline"
    return result, orchestrator


def plan_reply(*descriptions: str) -> str:
    tasks = ", ".join(
        f'{{"id": "{index}", "description": "{description}",'
        ' "files": ["src/app.py"], "dependencies": []}'
        for index, description in enumerate(descriptions, 1)
    )
    return f'{{"tasks": [{tasks}], "rationale": "one slice at a time"}}'


def file_reply(path: str, body: str) -> str:
    return f"Adding {path}.\n\n<<<FILE: {path}>>>\n{body}\n<<<END FILE>>>\n"


@pytest.fixture
def python_workspace(tmp_workspace: Path) -> Path:
    """A workspace the tester recognises as a Python project."""
    (tmp_workspace / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_workspace / "src").mkdir()
    return tmp_workspace


async def test_a_full_cycle_plans_writes_files_and_reports_success(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_workspace: Path,
    subprocess_stub: SubprocessStub,
    event_recorder: EventRecorder,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    subprocess_stub.queue(ProcessOutcome.of(0, stdout="1 passed in 0.01s"))
    client = make_fake_llm(
        [
            f"Here is the plan:\n```json\n{plan_reply('implement the parser')}\n```",
            file_reply("src/app.py", "def parse(text):\n    return text.split()"),
            '{"passed": true, "summary": "1 passed", "failures": []}',
        ]
    )

    result, orchestrator = await run_loop(client, "add a parser", event_recorder, bounded)

    assert result.stopped_reason == "goal-complete"
    assert result.passed is True
    assert result.tasks_completed == 1
    assert (python_workspace / "src" / "app.py").read_text() == (
        "def parse(text):\n    return text.split()\n"
    )
    assert [task.status for task in orchestrator.tasks] == ["completed"]
    assert event_recorder.names[0] == "run.start"
    assert event_recorder.names[-1] == "run.done"
    assert event_recorder.count("plan") == 1
    assert event_recorder.count("task.done") == 1


async def test_the_real_agents_are_driven_with_their_own_prompts_and_models(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_workspace: Path,
    subprocess_stub: SubprocessStub,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    subprocess_stub.configure(returncode=0, stdout="1 passed")
    client = make_fake_llm(
        [
            plan_reply("implement the parser"),
            file_reply("src/app.py", "x = 1"),
            '{"passed": true, "summary": "fine"}',
        ]
    )

    await run_loop(client, "add a parser", None, bounded)

    settings = client.settings
    assert client.systems[0] == PLANNER_PROMPT
    assert client.models[0] == settings.planner_model
    assert client.systems[1] == WORKER_PROMPT
    assert client.models[1] == settings.worker_model


async def test_a_failing_test_run_feeds_the_failure_back_into_the_next_plan(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_workspace: Path,
    subprocess_stub: SubprocessStub,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    subprocess_stub.queue(
        ProcessOutcome.of(1, stdout="FAILED tests/test_app.py::test_parse - AssertionError"),
        ProcessOutcome.of(0, stdout="1 passed"),
    )
    client = make_fake_llm(
        [
            plan_reply("implement the parser"),
            file_reply("src/app.py", "def parse(text):\n    return text.split()"),
            '{"passed": false, "summary": "test_parse failed"}',
            plan_reply("fix the parser"),
            file_reply("src/app.py", "def parse(text):\n    return text.strip()"),
            '{"passed": true, "summary": "green"}',
        ]
    )

    result, _ = await run_loop(client, "add a parser", None, bounded)

    assert result.cycles == 2, "the loop must come back with the failure"
    assert result.stopped_reason == "goal-complete"
    assert result.passed is True
    # per cycle the agents call the model in this order: planner, worker, tester
    assert client.call_count == 6
    second_plan_prompt = client.prompts[3]
    assert second_plan_prompt.startswith("Goal: add a parser")
    assert "Current focus:" in second_plan_prompt
    assert "test_parse" in second_plan_prompt


async def test_a_worker_refused_every_path_still_reports_what_happened(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_workspace: Path,
    tmp_path: Path,
    subprocess_stub: SubprocessStub,
    event_recorder: EventRecorder,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    """The security boundary holds end to end, and the refusal is not silently dropped."""
    subprocess_stub.configure(returncode=0, stdout="1 passed")
    client = make_fake_llm(
        [
            plan_reply("write outside the workspace"),
            file_reply("../escaped.py", "owned = True"),
            '{"passed": true, "summary": "fine"}',
        ]
    )

    result, orchestrator = await run_loop(client, "escape the workspace", event_recorder, bounded)

    assert not (tmp_path / "escaped.py").exists()
    assert "outside the workspace" in str(orchestrator.tasks[0].result)
    # WorkerAgent reports refusals in its summary instead of raising, so the orchestrator counts
    # the task as completed; the refusal is only visible in the summary and the task.done event.
    assert event_recorder.payloads("task.done")[0]["task_id"] == "1"
    assert result.tasks_completed == 1


async def test_an_unusable_plan_reply_stalls_the_run_without_crashing(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_workspace: Path,
    subprocess_stub: SubprocessStub,
    event_recorder: EventRecorder,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    subprocess_stub.configure(returncode=0, stdout="1 passed")
    client = make_fake_llm(["I am not going to answer with JSON."] * 3)

    result, orchestrator = await run_loop(client, "do something", event_recorder, bounded)

    assert result.stopped_reason == "stalled"
    assert result.passed is False
    assert orchestrator.tasks == []
    assert event_recorder.count("task.start") == 0


async def test_a_missing_llm_reply_turns_into_a_stall_not_a_crash(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_workspace: Path,
    subprocess_stub: SubprocessStub,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    subprocess_stub.configure(returncode=0, stdout="1 passed")
    client = make_fake_llm([RuntimeError("no API key configured")] * 3)

    result, _ = await run_loop(client, "do something", None, bounded)

    assert result.stopped_reason == "stalled"
    assert result.passed is False


async def test_the_loop_runs_in_the_configured_workspace(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_workspace: Path,
    subprocess_stub: SubprocessStub,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    subprocess_stub.configure(returncode=0, stdout="1 passed")
    client = make_fake_llm(
        [
            plan_reply("write a file"),
            file_reply("nested/deep/file.txt", "hello"),
            '{"passed": true, "summary": "fine"}',
        ]
    )

    await run_loop(client, "write a file", None, bounded)

    assert (python_workspace / "nested" / "deep" / "file.txt").read_text() == "hello\n"
    assert subprocess_stub.kwargs[0]["cwd"] == str(python_workspace)
