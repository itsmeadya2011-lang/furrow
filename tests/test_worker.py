"""Behavioural tests for the worker agent - the component that writes files.

A fake ``LLMClient`` records every ``complete`` call and stubs the filesystem
helpers, so nothing here reaches a network, a real model or the real disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from furrow.agents.worker import (
    CONTEXT_FOOTER,
    CONTEXT_HEADER,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_FILES,
    TRUNCATION_NOTICE,
    WorkerAgent,
)
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

SOURCE = "def add(a, b):\n    return a + b\n"


def make_task(*files: str, description: str = "add a docstring") -> TaskModel:
    return TaskModel(id="1", description=description, files=list(files))


def envelope(*edits: tuple[str, str], summary: str = "did the thing") -> dict[str, Any]:
    return {
        "summary": summary,
        "edits": [{"path": path, "content": content} for path, content in edits],
    }


class FakeClient(LLMClient):
    """A real ``LLMClient`` with scripted replies and an in-memory filesystem."""

    def __init__(
        self,
        reply: str,
        files: dict[str, str] | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings=settings or Settings())
        self.reply = reply
        self.files: dict[str, str] = dict(files or {})
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.models: list[str | None] = []
        self.writes: list[tuple[str, str]] = []
        self.reads: list[str] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        self.models.append(model)
        return self.reply

    async def read_file(self, path: str | Path) -> str:
        self.reads.append(str(path))
        try:
            return self.files[str(path)]
        except KeyError:
            raise FileNotFoundError(path) from None

    async def write_file(self, path: str | Path, content: str) -> None:
        self.writes.append((str(path), content))
        self.files[str(path)] = content


def fenced(payload: dict[str, Any]) -> str:
    return f"Here you go:\n```json\n{json.dumps(payload)}\n```\nHope that helps."


# --- prompt assembly -------------------------------------------------------


async def test_prompt_contains_current_file_contents() -> None:
    client = FakeClient(fenced(envelope(("src/a.py", "print(1)\n"))), files={"src/a.py": SOURCE})
    agent = WorkerAgent(task=make_task("src/a.py"), client=client)

    await agent.run()

    assert len(client.prompts) == 1
    prompt = client.prompts[0]
    assert SOURCE in prompt
    assert "src/a.py" in prompt
    assert make_task("src/a.py").description in prompt
    assert prompt.index(CONTEXT_HEADER) < prompt.index(SOURCE) < prompt.index(CONTEXT_FOOTER)
    assert client.reads == ["src/a.py"]
    # the worker model, not the global default, is what gets asked
    assert client.models == [client.settings.worker_model]
    assert client.systems and "JSON" in client.systems[0]


async def test_unreadable_files_are_reported_not_fatal() -> None:
    client = FakeClient(fenced(envelope()), files={})
    agent = WorkerAgent(task=make_task("missing.py", "there.py"), client=client)

    result = await agent.run()

    assert "could not be read" in client.prompts[0]
    assert "missing.py" in client.prompts[0]
    assert result.endswith("Wrote: (none)")


# --- applying edits --------------------------------------------------------


async def test_fenced_envelope_with_one_edit_writes_once() -> None:
    new_body = "def add(a, b):\n    '''Add.'''\n    return a + b\n"
    client = FakeClient(fenced(envelope(("src/a.py", new_body))), files={"src/a.py": SOURCE})
    agent = WorkerAgent(task=make_task("src/a.py"), client=client)

    result = await agent.run()

    assert client.writes == [("src/a.py", new_body)]
    assert agent.written_files == ["src/a.py"]
    assert client.files["src/a.py"] == new_body
    assert result == "did the thing\nWrote: src/a.py"


async def test_result_string_lists_every_written_path() -> None:
    client = FakeClient(
        fenced(envelope(("a.py", "a\n"), ("b.py", "b\n"), summary="two files")),
        files={"a.py": "", "b.py": ""},
    )
    agent = WorkerAgent(task=make_task("a.py", "b.py"), client=client)

    result = await agent.run()

    assert [path for path, _ in client.writes] == ["a.py", "b.py"]
    assert result == "two files\nWrote: a.py, b.py"
    assert agent.written_files == ["a.py", "b.py"]


async def test_empty_edits_array_touches_nothing() -> None:
    client = FakeClient(fenced(envelope(summary="nothing to do")), files={"a.py": SOURCE})
    agent = WorkerAgent(task=make_task("a.py"), client=client)

    result = await agent.run()

    assert client.writes == []
    assert agent.written_files == []
    assert result == "nothing to do\nWrote: (none)"


async def test_summary_falls_back_to_placeholder() -> None:
    client = FakeClient('{"edits": []}', files={"a.py": SOURCE})
    agent = WorkerAgent(task=make_task("a.py"), client=client)

    result = await agent.run()

    assert result.endswith("Wrote: (none)")
    assert result.splitlines()[0] == "worker returned edits without a summary"


# --- dry run ---------------------------------------------------------------


async def test_dry_run_plans_but_never_writes() -> None:
    new_body = "brand new\n"
    client = FakeClient(fenced(envelope(("a.py", new_body))), files={"a.py": SOURCE})
    agent = WorkerAgent(task=make_task("a.py"), client=client, dry_run=True)

    result = await agent.run()

    assert client.writes == []
    assert agent.written_files == []
    assert client.files["a.py"] == SOURCE  # untouched on disk
    assert result == "did the thing\nWrote: a.py"  # the plan is still reported


# --- validation ------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "I could not do that, sorry.",  # no JSON at all
        '["not", "an", "object"]',  # valid JSON, wrong shape
        '{"summary": "x", "edits": "a.py"}',  # edits is not a list
        '{"summary": "x", "edits": ["src/a.py"]}',  # edit is not an object
        '{"summary": "x", "edits": [{"content": "hi"}]}',  # no path
        '{"summary": "x", "edits": [{"path": 7, "content": "hi"}]}',  # path not a str
        '{"summary": "x", "edits": [{"path": "a.py"}]}',  # no content
        '{"summary": "x", "edits": [{"path": "a.py", "content": 12}]}',  # content not a str
        '{"summary": "x", "edits": [{"path": "a.py", "content": ]}',  # unparseable payload
    ],
)
async def test_malformed_envelope_raises_value_error(reply: str) -> None:
    client = FakeClient(reply, files={"a.py": SOURCE})
    agent = WorkerAgent(task=make_task("a.py"), client=client)

    with pytest.raises(ValueError) as excinfo:
        await agent.run()

    assert reply in str(excinfo.value)
    assert client.writes == []


# --- context bounds --------------------------------------------------------


async def test_file_count_bound_is_enforced() -> None:
    many = [f"f{index}.py" for index in range(MAX_CONTEXT_FILES + 5)]
    client = FakeClient(fenced(envelope()), files={name: SOURCE for name in many})
    agent = WorkerAgent(task=make_task(*many), client=client)

    await agent.run()

    assert client.reads == many[:MAX_CONTEXT_FILES]
    prompt = client.prompts[0]
    assert "omitted to stay within the 20 file limit" in prompt
    assert f"f{MAX_CONTEXT_FILES}.py" in prompt  # named as omitted
    assert SOURCE in prompt


async def test_size_bound_truncates_the_context_block() -> None:
    big = "# padding\n" * (MAX_CONTEXT_CHARS // 2)
    assert len(big) > MAX_CONTEXT_CHARS
    client = FakeClient(fenced(envelope()), files={"big.py": big})
    agent = WorkerAgent(task=make_task("big.py"), client=client)

    await agent.run()

    context = client.prompts[0]
    assert TRUNCATION_NOTICE.strip() in context
    assert "truncated" in context
    # the block itself is clamped, notes and notice included
    header_at = context.index(CONTEXT_HEADER)
    footer_gone = CONTEXT_FOOTER not in context[header_at:]
    assert footer_gone
    block = context[header_at:]
    assert len(block) <= MAX_CONTEXT_CHARS + len("\n")


async def test_small_context_is_left_alone() -> None:
    client = FakeClient(fenced(envelope()), files={"a.py": SOURCE})
    agent = WorkerAgent(task=make_task("a.py"), client=client)

    await agent.run()

    prompt = client.prompts[0]
    assert TRUNCATION_NOTICE not in prompt
    assert CONTEXT_FOOTER in prompt


async def test_no_assigned_files_still_asks_the_model() -> None:
    client = FakeClient(fenced(envelope(("new.py", "created\n"))))
    agent = WorkerAgent(task=make_task(), client=client)

    result = await agent.run()

    assert client.reads == []
    assert "none assigned" in client.prompts[0]
    assert client.writes == [("new.py", "created\n")]
    assert result.endswith("Wrote: new.py")


# --- construction compatibility -------------------------------------------


def test_accepts_the_orchestrators_call_forms() -> None:
    settings = Settings()
    client = FakeClient("{}", settings=settings)
    task = make_task("a.py")

    positional = WorkerAgent(task, client, settings)
    assert positional.client is client
    assert positional.dry_run is False
    assert positional.written_files == []

    # no client, explicit settings -> a real LLMClient bound to those settings
    assert WorkerAgent(task=task, settings=settings).client.settings is settings
    # no client, no settings -> global fallback, never ``settings=None``
    assert WorkerAgent(task=task).client.settings is not None
