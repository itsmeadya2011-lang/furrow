"""Tests for :class:`furrow.agents.worker.WorkerAgent`.

The worker is the only agent that takes a task instead of a goal, so the tests focus
on the prompt it builds from that task (description plus the file allow-list, with a
fallback when no files are named) and on the fact that it hands the raw report text
back untouched.
"""

from __future__ import annotations

import pytest

from furrow.agents.prompts import WORKER_PROMPT
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from tests.conftest import FakeLLM

DESCRIPTION = "Implement the JWT helpers"
REPORT = (
    '{"summary": "Added the JWT helpers.", "files_changed": ["src/auth.py"], '
    '"status": "completed", "blockers": []}'
)

TASK = TaskModel(
    id="1",
    description=DESCRIPTION,
    files=["src/auth.py", "tests/test_auth.py"],
    dependencies=[],
)


def _worker(
    task: TaskModel,
    response: str = REPORT,
    settings: Settings | None = None,
) -> tuple[WorkerAgent, FakeLLM]:
    """A worker plus the fake client it was given, for prompt assertions."""
    fake_llm = FakeLLM(response=response)
    return WorkerAgent(task, client=fake_llm, settings=settings), fake_llm


async def test_run_prompt_contains_the_task_description(settings: Settings) -> None:
    agent, fake_llm = _worker(TASK, settings=settings)

    await agent.run()

    prompt = fake_llm.calls[0][0]
    assert f"Task: {DESCRIPTION}" in prompt


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        pytest.param(
            ["src/auth.py", "tests/test_auth.py"],
            "src/auth.py, tests/test_auth.py",
            id="two-files",
        ),
        pytest.param(["src/auth.py"], "src/auth.py", id="one-file"),
        pytest.param([], "any", id="no-files-falls-back-to-any"),
    ],
)
async def test_run_prompt_renders_the_file_list(files: list[str], expected: str, settings: Settings) -> None:
    task = TaskModel(id="1", description=DESCRIPTION, files=files)
    agent, fake_llm = _worker(task, settings=settings)

    report = await agent.run()

    prompt = fake_llm.calls[0][0]
    assert f"Files to touch: {expected}\n" in prompt
    assert report == REPORT


async def test_run_sends_the_worker_persona_as_the_system_message(settings: Settings) -> None:
    agent, fake_llm = _worker(TASK, settings=settings)

    await agent.run()

    prompt, system, _model = fake_llm.calls[0]
    assert len(fake_llm.calls) == 1
    assert system == WORKER_PROMPT
    assert WORKER_PROMPT not in prompt


async def test_run_returns_the_raw_report_text(settings: Settings) -> None:
    """The report is the worker's verbatim text: no parsing, no JSON requirement."""
    report = "I could not finish: the task description was too vague."
    agent, _fake_llm = _worker(TASK, response=report, settings=settings)

    result = await agent.run()

    assert isinstance(result, str)
    assert result == report


async def test_run_requests_the_configured_worker_model(settings: Settings) -> None:
    agent, fake_llm = _worker(TASK, settings=settings)

    await agent.run()

    assert fake_llm.calls[0][2] == settings.worker_model
    assert agent.model_attr == "worker_model"


def test_worker_constructs_standalone() -> None:
    """Constructing a worker must not need a task's client to exist yet."""
    agent = WorkerAgent(TASK)

    assert agent.task is TASK
    assert agent.settings is not None
    assert agent.client is not None
    assert agent.model_attr == "worker_model"


async def test_run_leaves_the_task_untouched(settings: Settings) -> None:
    """The task is shared state across cycles, so building the prompt must not mutate it."""
    task = TaskModel(id="7", description="Tidy the changelog")
    agent, _fake_llm = _worker(task, settings=settings)

    await agent.run()

    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None
