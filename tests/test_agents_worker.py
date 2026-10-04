"""Behavioural tests for :mod:`furrow.agents.worker`.

Every test drives :class:`~furrow.agents.worker.WorkerAgent` through the
``fake_client`` fixture from ``tests/conftest.py``, so nothing here touches the
network: no real :class:`~furrow.llm.LLMClient` is ever constructed, and the
workspace and settings come from the hermetic ``tmp_workspace`` /
``make_settings`` fixtures.

Three properties get the most attention, because each one is a contract the
orchestrator relies on:

* ``run()`` never raises for an ordinary failure. A provider error, a
  message-less exception or a malformed tool call comes back as
  ``WorkerResult(ok=False, error=...)`` instead of taking the run down.
* ``files_changed`` reports only the ``path`` arguments of the two mutating
  tools (``write_file`` / ``edit_file``), de-duplicated and sorted.
* :class:`asyncio.CancelledError` PROPAGATES. It is a shutdown signal, not a
  task failure; swallowing it would leave the orchestrator's ``gather`` hanging
  forever, so the blanket ``except Exception`` cannot see it and the explicit
  re-raise clause is asserted directly.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from furrow.agents.prompts import FIX_PROMPT, WORKER_PROMPT, WORKER_SYSTEM
from furrow.agents.worker import TRUNCATION_NOTE, WorkerAgent, WorkerResult
from furrow.config import Settings, TaskModel, TaskStatus
from furrow.llm import CompletionResult, ToolCall

#: Upper bound for "the worker should already be suspended here". A generous
#: value: it only ever trips on a hang, never on a slow machine.
TIMEOUT = 5.0

#: The placeholder :mod:`furrow.agents.worker` substitutes when a task names no
#: files, so the prompt still tells the worker what to do.
NO_FILES_PLACEHOLDER = "(not specified - choose the files the task needs)"

TASK_ID = "task-1"
DESCRIPTION = "Add a JSON parser to furrow/parsing.py"
TASK_FILES = ["furrow/parsing.py", "furrow/config.py"]


# --- helpers ---


def _task(**overrides: Any) -> TaskModel:
    """A :class:`~furrow.config.TaskModel` with overrides applied."""
    values: dict[str, Any] = {
        "id": TASK_ID,
        "description": DESCRIPTION,
        "files": list(TASK_FILES),
    }
    values.update(overrides)
    return TaskModel(**values)


@dataclass(slots=True)
class _ArgumentlessCall:
    """A tool-call stand-in with no ``arguments`` attribute at all.

    ``ToolCall`` declares ``arguments`` as a required field, so only a
    hand-built object can reach the ``getattr(call, "arguments", None)`` branch
    of ``_changed_files``. Everything the worker reads off a call (``name``,
    ``arguments``) is present here.
    """

    name: str


class _BlockingClient:
    """Minimal stand-in whose ``complete_with_tools`` never returns.

    Used to cancel the worker while a provider call is genuinely in flight,
    which is the situation ``asyncio.Task.cancel()`` actually produces during
    shutdown. ``WorkerAgent`` only requires the ``complete_with_tools``
    coroutine on the client it is handed.
    """

    def __init__(self, entered: asyncio.Event) -> None:
        self.entered = entered
        self.prompts: list[str] = []

    async def complete_with_tools(self, prompt: str, **_kwargs: Any) -> CompletionResult:
        """Record the prompt, signal that the call started, then hang."""
        self.prompts.append(prompt)
        self.entered.set()
        await asyncio.Event().wait()  # pragma: no cover - never completes
        raise AssertionError("unreachable: the wait above cannot finish")  # pragma: no cover


# --- files_changed / tool_calls ---


async def test_write_and_edit_calls_report_sorted_unique_files(
    make_settings: Callable[..., Settings],
    fake_client: Callable[..., Any],
) -> None:
    """The two mutating tools contribute their ``path``, sorted, one per file."""
    settings = make_settings()
    client = fake_client(
        settings,
        tool_result=CompletionResult(
            text="Added the parser and its config hook.",
            tool_calls=[
                ToolCall(
                    name="write_file",
                    arguments={"path": "src/zeta.py", "content": "z = 1\n"},
                ),
                ToolCall(
                    name="edit_file",
                    arguments={
                        "path": "src/alpha.py",
                        "old": "a = 1",
                        "new": "a = 2",
                    },
                ),
            ],
            iterations=2,
        ),
    )

    result = await WorkerAgent(_task(), client, settings=settings).run()

    assert result.ok is True
    assert result.tool_calls == 2
    # Sorted, not call order: "src/alpha.py" was the second call.
    assert result.files_changed == ["src/alpha.py", "src/zeta.py"]
    assert result.summary == "Added the parser and its config hook."
    assert result.iterations == 2
    assert result.truncated is False
    assert result.error is None


async def test_files_changed_dedupes_paths_and_skips_read_only_tools(
    make_settings: Callable[..., Settings],
    fake_client: Callable[..., Any],
) -> None:
    """Repeated paths collapse; ``read_file`` / ``list_files`` never count."""
    settings = make_settings()
    client = fake_client(
        settings,
        tool_result=CompletionResult(
            text="Done.",
            tool_calls=[
                ToolCall(name="read_file", arguments={"path": "src/untouched.py"}),
                ToolCall(name="write_file", arguments={"path": "src/b.py", "content": "one"}),
                ToolCall(name="write_file", arguments={"path": "src/b.py", "content": "two"}),
                ToolCall(name="edit_file", arguments={"path": "src/a.py", "old": "x", "new": "y"}),
                ToolCall(name="list_files", arguments={"path": "src"}),
            ],
            iterations=3,
        ),
    )

    result = await WorkerAgent(_task(), client, settings=settings).run()

    # Every call is counted, including the duplicate and the read-only ones.
    assert result.tool_calls == 5
    assert result.files_changed == ["src/a.py", "src/b.py"]


async def test_files_changed_skips_a_path_that_is_not_a_usable_string(
    make_settings: Callable[..., Settings],
    fake_client: Callable[..., Any],
) -> None:
    """A missing, blank or non-string ``path`` is dropped, not recorded."""
    settings = make_settings()
    client = fake_client(
        settings,
        tool_result=CompletionResult(
            text="Done.",
            tool_calls=[
                ToolCall(name="write_file", arguments={}),
                ToolCall(name="write_file", arguments={"path": "   "}),
                ToolCall(name="write_file", arguments={"path": 42}),
                ToolCall(name="edit_file", arguments={"path": None}),
                ToolCall(name="write_file", arguments={"path": "src/real.py", "content": "x"}),
            ],
        ),
    )

    result = await WorkerAgent(_task(), client, settings=settings).run()

    assert result.ok is True
    assert result.tool_calls == 5
    assert result.files_changed == ["src/real.py"]


async def test_malformed_tool_arguments_do_not_crash_the_worker(
    make_settings: Callable[..., Settings],
    fake_client: Callable[..., Any],
) -> None:
    """Missing / ``None`` / non-dict ``arguments`` are skipped, never crash.

    ``_changed_files`` guards with ``getattr(call, "arguments", None)`` and an
    ``isinstance(arguments, dict)`` check. The ``None`` and non-dict cases are
    reachable with a real :class:`~furrow.llm.ToolCall` (the dataclass does not
    type-check its fields); the attribute-missing case needs a stand-in.
    """
    settings = make_settings()
    client = fake_client(
        settings,
        tool_result=CompletionResult(
            text="Worked around the bad arguments.",
            tool_calls=[
                _ArgumentlessCall(name="write_file"),
                ToolCall(name="write_file", arguments=None),
                ToolCall(name="edit_file", arguments="path=src/a.py"),
                ToolCall(name="write_file", arguments=["src/a.py"]),
                ToolCall(name="write_file", arguments={"path": "src/ok.py", "content": "x"}),
            ],
        ),
    )

    result = await WorkerAgent(_task(), client, settings=settings).run()

    assert result.ok is True
    assert result.tool_calls == 5
    assert result.files_changed == ["src/ok.py"]
    assert result.summary == "Worked around the bad arguments."


async def test_a_run_without_tool_calls_reports_nothing_changed(
    make_settings: Callable[..., Settings],
    fake_client: Callable[..., Any],
) -> None:
    """The default fake completion (no tools) yields an empty change set."""
    settings = make_settings()
    client = fake_client(settings)

    result = await WorkerAgent(_task(), client, settings=settings).run()

    assert result.ok is True
    assert result.tool_calls == 0
    assert result.files_changed == []
    assert result.truncated is False
    assert result.error is None


# --- failures never escape run() ---


async def test_a_provider_error_comes_back_as_a_failed_result(
    make_settings: Callable[..., Settings],
    fake_client: Callable[..., Any],
) -> None:
    """Nothing escapes ``run()``; the message lands on ``WorkerResult.error``."""
    settings = make_settings()
    client = fake_client(settings, raises=RuntimeError("provider exploded"))

    result = await WorkerAgent(_task(), client, settings=settings).run()

    assert isinstance(result, WorkerResult)
    assert result.ok is False
    assert result.error == "provider exploded"
    assert result.summary == ""
    assert result.tool_calls == 0
    assert result.files_changed == []
    assert result.iterations == 0
    assert result.truncated is False
    # The request was still attempted, so the failure is attributable.
    assert len(client.calls) == 1
    assert client.calls[0].prompt.startswith("Attempt 1")


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        pytest.param(TimeoutError(), "TimeoutError", id="timeout"),
        pytest.param(MemoryError(), "MemoryError", id="memory"),
        pytest.param(ValueError(""), "ValueError", id="empty-message"),
    ],
)
async def test_a_message_less_exception_reports_its_type(
    make_settings: Callable[..., Settings],
    fake_client: Callable[..., Any],
    exc: BaseException,
    expected: str,
) -> None:
    """``str(exc)`` is empty for these; the type name keeps the failure readable.

    An empty ``error`` would reach the reporter and the CLI as a blank string,
    so the worker falls back to ``type(exc).__name__``.
    """
    assert str(exc) == ""
    settings = make_settings()
    client = fake_client(settings, raises=exc)

    result = await WorkerAgent(_task(), client, settings=settings).run()

    assert result.ok is False
    assert result.error == expected


# --- prompt selection ---


async def test_first_attempt_uses_the_worker_prompt(
    make_settings: Callable[..., Settings],
    fake_client: Callable[..., Any],
) -> None:
    """``attempt=1`` renders :data:`WORKER_PROMPT` with the task's files."""
    settings = make_settings()
    client = fake_client(settings)
    worker = WorkerAgent(_task(), client, settings=settings, attempt=1)

    assert await worker.run() is not None

    call = client.calls[0]
    assert call.prompt == WORKER_PROMPT.format(
        description=DESCRIPTION,
        files="furrow/parsing.py, furrow/config.py",
        attempt=1,
        feedback="(none)",
    )
    assert call.system == WORKER_SYSTEM
    assert call.model == settings.model_for("worker")
    assert call.max_iterations == settings.worker_max_iterations
    assert call.tools is worker.tools


async def test_a_task_without_files_gets_a_placeholder(
    make_settings: Callable[..., Settings],
    fake_client: Callable[..., Any],
) -> None:
    """An empty ``files`` list still renders a usable prompt."""
    settings = make_settings()
    client = fake_client(settings)
    worker = WorkerAgent(_task(files=[]), client, settings=settings)

    await worker.run()

    assert client.calls[0].prompt == WORKER_PROMPT.format(
        description=DESCRIPTION,
        files=NO_FILES_PLACEHOLDER,
        attempt=1,
        feedback="(none)",
    )
    assert NO_FILES_PLACEHOLDER in client.calls[0].prompt


async def test_a_repair_attempt_uses_the_fix_prompt(
    make_settings: Callable[..., Settings],
    fake_client: Callable[..., Any],
) -> None:
    """``attempt=2`` renders :data:`FIX_PROMPT`, keeping the goal in view."""
    settings = make_settings()
    client = fake_client(settings)
    feedback = "test_parse_json gave up: expected a list, got a dict"
    worker = WorkerAgent(_task(), client, settings=settings, attempt=2, feedback=feedback)

    await worker.run()

    prompt = client.calls[0].prompt
    assert prompt == FIX_PROMPT.format(
        goal=DESCRIPTION,
        failures=feedback,
        attempt=2,
    )
    # The repair prompt is a different template, not the worker one.
    assert "Feedback from a previous attempt" not in prompt


@pytest.mark.parametrize("attempt", [1, 3])
async def test_feedback_is_forwarded_into_the_prompt(
    make_settings: Callable[..., Settings],
    fake_client: Callable[..., Any],
    attempt: int,
) -> None:
    """Feedback reaches the prompt on the first attempt and on a retry alike."""
    settings = make_settings()
    client = fake_client(settings)
    feedback = "the previous attempt said it could not find the file"
    worker = WorkerAgent(_task(), client, settings=settings, attempt=attempt, feedback=feedback)

    await worker.run()

    prompt = client.calls[0].prompt
    assert feedback in prompt
    assert prompt.startswith(f"Attempt {attempt}\n")
    # No template placeholder survives rendering.
    assert "{" not in prompt and "}" not in prompt


# --- cancellation propagates ---


def test_cancelled_error_is_outside_the_scope_of_except_exception() -> None:
    """Why the explicit re-raise clause is needed at all.

    ``asyncio.CancelledError`` derives from ``BaseException``, so the blanket
    ``except Exception`` in ``run()`` cannot see it. If that ever changed, a
    cancelled worker would return a ``WorkerResult`` instead of propagating and
    shutdown would hang.
    """
    assert issubclass(asyncio.CancelledError, BaseException)
    assert not issubclass(asyncio.CancelledError, Exception)


async def test_cancellation_propagates_from_an_in_flight_call(
    make_settings: Callable[..., Settings],
) -> None:
    """Cancelling a suspended ``run()`` ends the task cancelled, not resolved.

    The coroutine is suspended inside ``complete_with_tools`` when the cancel
    arrives, so the ``CancelledError`` is raised exactly where the real provider
    await sits. If ``run()`` swallowed it, ``await task`` would return a
    ``WorkerResult`` and ``task.cancelled()`` would be False.
    """
    settings = make_settings()
    entered = asyncio.Event()
    client = _BlockingClient(entered)
    worker = WorkerAgent(_task(), client, settings=settings)

    task = asyncio.create_task(worker.run())
    try:
        await asyncio.wait_for(entered.wait(), timeout=TIMEOUT)
        assert len(client.prompts) == 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() is True
    finally:
        if not task.done():  # pragma: no cover - only on a failure path
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_cancelled_error_raised_by_the_client_propagates(
    make_settings: Callable[..., Settings],
    fake_client: Callable[..., Any],
) -> None:
    """A cancellation delivered at the await point is re-raised, not captured."""
    settings = make_settings()
    client = fake_client(settings, raises=asyncio.CancelledError())
    worker = WorkerAgent(_task(), client, settings=settings)

    with pytest.raises(asyncio.CancelledError):
        await worker.run()


# --- truncation ---


async def test_a_truncated_completion_is_flagged_and_noted(
    make_settings: Callable[..., Settings],
    fake_client: Callable[..., Any],
) -> None:
    """``truncated`` reaches the result and the summary carries the warning."""
    settings = make_settings()
    client = fake_client(
        settings,
        tool_result=CompletionResult(
            text="Started the edit but ran out of room.",
            tool_calls=[ToolCall(name="write_file", arguments={"path": "src/a.py"})],
            truncated=True,
        ),
    )

    result = await WorkerAgent(_task(), client, settings=settings).run()

    assert result.truncated is True
    assert result.ok is True
    assert result.files_changed == ["src/a.py"]
    assert result.summary.startswith("Started the edit but ran out of room.")
    assert TRUNCATION_NOTE in result.summary
    assert result.summary == f"Started the edit but ran out of room.\n\n{TRUNCATION_NOTE}"


async def test_an_untruncated_summary_gets_no_truncation_note(
    make_settings: Callable[..., Settings],
    fake_client: Callable[..., Any],
) -> None:
    """The note is added only when the completion reports truncation."""
    settings = make_settings()
    client = fake_client(
        settings,
        tool_result=CompletionResult(text="All done.", truncated=False),
    )

    result = await WorkerAgent(_task(), client, settings=settings).run()

    assert result.truncated is False
    assert result.summary == "All done."
    assert TRUNCATION_NOTE not in result.summary


async def test_a_truncated_empty_summary_is_just_the_note(
    make_settings: Callable[..., Settings],
    fake_client: Callable[..., Any],
) -> None:
    """With no text, the summary is the note alone rather than a blank prefix."""
    settings = make_settings()
    client = fake_client(settings, tool_result=CompletionResult(text="", truncated=True))

    result = await WorkerAgent(_task(), client, settings=settings).run()

    assert result.summary == TRUNCATION_NOTE
    assert result.truncated is True


async def test_summary_whitespace_is_stripped(
    make_settings: Callable[..., Settings],
    fake_client: Callable[..., Any],
) -> None:
    """A padded model reply does not leak its padding into the report."""
    settings = make_settings()
    client = fake_client(settings, tool_result=CompletionResult(text="\n\n  Padded.  \n"))

    result = await WorkerAgent(_task(), client, settings=settings).run()

    assert result.summary == "Padded."


# --- construction and side effects ---


def test_default_tools_are_built_from_the_given_settings(
    make_settings: Callable[..., Settings],
    tmp_workspace: Any,
) -> None:
    """``tools=None`` sandboxes the worker to the configured workspace."""
    settings = make_settings()
    worker = WorkerAgent(_task(), _BlockingClient(asyncio.Event()), settings=settings)

    assert worker.tools.root == tmp_workspace.resolve()
    assert worker.tools.root.is_absolute()


def test_task_and_tools_are_exposed_read_only(
    make_settings: Callable[..., Settings],
) -> None:
    """``task`` / ``tools`` report what the worker was built with."""
    settings = make_settings()
    task = _task()
    worker = WorkerAgent(task, _BlockingClient(asyncio.Event()), settings=settings)

    assert worker.task is task
    assert worker.task.description == DESCRIPTION
    assert set(worker.tools.names()) >= {"read_file", "write_file", "edit_file"}


async def test_a_failed_run_does_not_mutate_the_task(
    make_settings: Callable[..., Settings],
    fake_client: Callable[..., Any],
) -> None:
    """Task state is the orchestrator's; the worker only reads the task."""
    settings = make_settings()
    task = _task()
    client = fake_client(settings, raises=RuntimeError("boom"))

    result = await WorkerAgent(task, client, settings=settings).run()

    assert result.ok is False
    assert task.status is TaskStatus.PENDING
    assert task.attempts == 0
    assert task.result is None
    assert task.error is None
    assert task.started_at is None
    assert task.finished_at is None


async def test_worker_result_to_dict_mirrors_every_field() -> None:
    """The JSON view used in event payloads is a faithful copy."""
    result = WorkerResult(
        ok=True,
        summary="Changed one file.",
        tool_calls=3,
        files_changed=["src/a.py"],
        iterations=4,
        truncated=True,
    )

    payload = result.to_dict()

    assert payload == {
        "ok": True,
        "summary": "Changed one file.",
        "tool_calls": 3,
        "files_changed": ["src/a.py"],
        "error": None,
        "iterations": 4,
        "truncated": True,
    }
    # A copy: mutating the payload cannot corrupt the result.
    payload["files_changed"].append("src/b.py")
    assert result.files_changed == ["src/a.py"]


def test_worker_result_defaults_for_a_failed_attempt() -> None:
    """A freshly constructed failure carries an empty, non-shared change list."""
    first = WorkerResult(ok=False, error="nope")
    second = WorkerResult(ok=False, error="still nope")

    assert first.summary == ""
    assert first.tool_calls == 0
    assert first.iterations == 0
    assert first.truncated is False
    assert first.files_changed == []
    first.files_changed.append("leak")
    assert second.files_changed == []
