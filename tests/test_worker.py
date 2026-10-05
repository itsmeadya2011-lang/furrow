"""Regression tests for :class:`furrow.agents.worker.WorkerAgent`.

The worker is the thinnest of the three agents: build a prompt from the task, call the model
that ``worker_model`` names, hand the answer back untouched. These tests pin all three.
"""

from __future__ import annotations

import asyncio

import pytest

from furrow.agents.prompts import WORKER_PROMPT
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel

from .conftest import StubLLMClient


def stub(summary: str = "summary", settings: Settings | None = None) -> StubLLMClient:
    return StubLLMClient(settings=settings, summaries=[summary])


def run(task: TaskModel, client: StubLLMClient | None = None, summary: str = "summary") -> str:
    async def scenario() -> str:
        return await WorkerAgent(task=task, client=client or stub(summary)).run()

    return asyncio.run(scenario())


def test_the_worker_returns_the_model_summary_verbatim() -> None:
    task = TaskModel(id="1", description="Implement the widget")

    assert run(task, summary="changed 3 files, all green") == "changed 3 files, all green"


def test_the_prompt_uses_the_shared_worker_instructions() -> None:
    client = stub()
    task = TaskModel(id="1", description="Implement the widget")

    run(task, client=client)

    assert client.worker_prompts[0].startswith(WORKER_PROMPT)


def test_the_prompt_names_the_task_description() -> None:
    client = stub()
    task = TaskModel(id="1", description="Implement the widget")

    run(task, client=client)

    assert client.worker_prompts[0] == (
        f"{WORKER_PROMPT}\n\nTask: Implement the widget\nFiles to touch: any\n"
    )


def test_the_prompt_lists_the_files_to_touch() -> None:
    client = stub()
    task = TaskModel(id="1", description="Do it", files=["src/a.py", "src/b.py", "tests/test_a.py"])

    run(task, client=client)

    assert client.worker_prompts[0] == (
        f"{WORKER_PROMPT}\n\nTask: Do it\nFiles to touch: src/a.py, src/b.py, tests/test_a.py\n"
    )


@pytest.mark.parametrize("files", [[], ["src/a.py"]])
def test_a_task_with_no_declared_files_says_any(files: list[str]) -> None:
    client = stub()
    task = TaskModel(id="1", description="Do it", files=files)

    run(task, client=client)

    if files:
        assert client.worker_prompts[0].endswith("Files to touch: src/a.py\n")
    else:
        assert client.worker_prompts[0].endswith("Files to touch: any\n")


def test_the_default_task_has_no_files() -> None:
    client = stub()

    run(TaskModel(id="1", description="Do it"), client=client)

    assert client.worker_prompts[0].endswith("Files to touch: any\n")


def test_the_configured_worker_model_is_requested() -> None:
    client = stub(settings=Settings(worker_model="worker-model-x"))
    task = TaskModel(id="1", description="Do it")

    run(task, client=client)

    assert client.calls[0].model == "worker-model-x"


def test_the_worker_never_falls_back_to_the_generic_model() -> None:
    settings = Settings(model="generic", worker_model="worker-specific")
    client = stub(settings=settings)
    task = TaskModel(id="1", description="Do it")

    run(task, client=client)

    assert client.calls[0].model == settings.worker_model
    assert client.calls[0].model != settings.model


def test_the_worker_sends_no_system_message() -> None:
    client = stub()
    task = TaskModel(id="1", description="Do it")

    run(task, client=client)

    assert client.calls[0].system == ""


def test_the_task_is_kept_on_the_agent() -> None:
    task = TaskModel(id="42", description="Do it")
    agent = WorkerAgent(task=task, client=stub())

    asyncio.run(agent.run())

    assert agent.task is task


def test_a_client_is_built_when_none_is_supplied() -> None:
    agent = WorkerAgent(task=TaskModel(id="1", description="Do it"))

    assert agent.client is not None
    assert agent.client.settings is not None


def test_each_run_makes_exactly_one_call() -> None:
    client = stub()
    task = TaskModel(id="1", description="Do it")

    run(task, client=client)

    assert len(client.calls) == 1
    assert len(client.worker_prompts) == 1
