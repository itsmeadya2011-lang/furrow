"""Regression tests for the ``furrow.agents.worker`` tool loop.

Self-contained on purpose: this module defines its own stub LLM client so it runs
without a shared ``conftest.py`` and without any network or subprocess access.

The stub does *real* filesystem work inside the per-test ``tmp_path`` workspace so
the path-escape guard in ``WorkerAgent._resolve`` is genuinely exercised. As a
safety net the stub refuses to touch anything outside the workspace even if that
guard ever regressed: an escaped write would mangle the test machine instead of
just failing an assertion, so ``StubClient.write_file`` records the attempt in
``escape_attempts`` and leaves the filesystem untouched.

Several tests below deliberately pin *lenient* behaviour that differs from what
the prompts in ``furrow.agents.worker`` ask the model for (see the "documented
deviations" section). Those are pinned, not endorsed, so a future fix shows up as
an intentional change instead of a silent one.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from furrow.agents.worker import (
    MAX_LIST_ENTRIES,
    MAX_READ_CHARS,
    MAX_TURNS,
    VALID_ACTIONS,
    WORKER_TOOL_PROMPT,
    WorkerAgent,
)
from furrow.config import TaskModel

WORKER_MODEL = "test-worker-model"

#: A well-formed terminal reply, reused as the second turn of most scripts.
DONE_REPLY = json.dumps({"action": "done", "summary": "finished the task"})

SENTINEL_TEXT = "do not touch\n"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _action(action: str, **fields: Any) -> str:
    """Serialize a single JSON action object the way a model would reply."""
    return json.dumps({"action": action, **fields})


def _task(description: str = "implement the thing", files: list[str] | None = None) -> TaskModel:
    return TaskModel(id="task-1", description=description, files=files or [])


def _inside(root: Path, target: Path) -> bool:
    return target.resolve().is_relative_to(root.resolve())


class StubClient:
    """Scripted stand-in for :class:`furrow.llm.LLMClient`."""

    def __init__(self, workspace: Path, replies: list[str], *, repeat_last: bool = False) -> None:
        self.settings = SimpleNamespace(workspace=workspace, worker_model=WORKER_MODEL)
        self.replies = list(replies)
        self.repeat_last = repeat_last
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.models: list[str | None] = []
        self.reads: list[Path] = []
        self.writes: list[tuple[Path, str]] = []
        self.lists: list[Path] = []
        self.escape_attempts: list[Path] = []

    @property
    def complete_count(self) -> int:
        return len(self.prompts)

    def _next_reply(self) -> str:
        index = len(self.prompts) - 1
        if index < len(self.replies):
            return self.replies[index]
        if self.repeat_last and self.replies:
            return self.replies[-1]
        raise AssertionError(
            f"worker requested reply #{index + 1} but only {len(self.replies)} were scripted"
        )

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        self.models.append(model)
        return self._next_reply()

    async def read_file(self, path: str | Path) -> str:
        target = Path(path)
        self.reads.append(target)
        # Mirrors aiofiles: a missing file raises rather than returning "".
        return target.read_text(encoding="utf-8")

    async def write_file(self, path: str | Path, content: str) -> None:
        target = Path(path)
        self.writes.append((target, content))
        if not _inside(Path(self.settings.workspace), target):
            # Safety net, see module docstring: record, never destroy.
            self.escape_attempts.append(target)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def list_files(self, directory: str | Path) -> list[str]:
        target = Path(directory)
        self.lists.append(target)
        if not target.is_dir():
            return []
        return sorted(f.relative_to(target).as_posix() for f in target.rglob("*") if f.is_file())


def _workspace(tmp_path: Path, name: str = "ws") -> Path:
    path = (tmp_path / name).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _worker(
    workspace: Path,
    replies: list[str],
    *,
    repeat_last: bool = False,
    task: TaskModel | None = None,
) -> tuple[WorkerAgent, StubClient]:
    stub = StubClient(workspace, replies, repeat_last=repeat_last)
    return WorkerAgent(task or _task(), client=stub), stub


def _stray_files(tmp_path: Path, workspace: Path) -> list[str]:
    """Every file under ``tmp_path`` that is lexically outside the workspace tree.

    Lexical on purpose: the in-workspace symlink used by the escape tests points
    out of the workspace and must not be counted as a stray file (its target's
    content is asserted separately).
    """
    return sorted(
        p.relative_to(tmp_path).as_posix()
        for p in tmp_path.rglob("*")
        if p.is_file() and not p.is_relative_to(workspace)
    )


def _escape_layout(tmp_path: Path) -> tuple[Path, Path]:
    """Workspace plus an outside sentinel and an escaping symlink inside it."""
    workspace = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text(SENTINEL_TEXT, encoding="utf-8")
    (workspace / "link.txt").symlink_to(sentinel)
    return workspace, sentinel


def _escape_paths(sentinel: Path) -> dict[str, str]:
    """Model-supplied paths that must all be refused by the guard."""
    return {
        "absolute-system-path": "/etc/passwd",
        "relative-traversal": "../outside.txt",
        "absolute-sibling-dir": str(sentinel.parent / "sibling.txt"),
        "symlink-out-of-workspace": "link.txt",
    }


ESCAPE_CASES = [
    "absolute-system-path",
    "relative-traversal",
    "absolute-sibling-dir",
    "symlink-out-of-workspace",
]


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #


async def test_happy_path_list_read_write_done(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "notes.txt").write_text("original note\n", encoding="utf-8")
    content = "print('hi')\n"
    worker, stub = _worker(
        workspace,
        [
            _action("list", path="."),
            _action("read", path="notes.txt"),
            _action("write", path="out/app.py", content=content),
            _action("done", summary="Added out/app.py."),
        ],
    )

    summary = await worker.run()

    # Exactly one turn per scripted reply, and the worker model on every call.
    assert stub.complete_count == 4
    assert stub.models == [WORKER_MODEL] * 4
    # Documented deviation: the worker never sends a system prompt.
    assert stub.systems == [""] * 4

    # Actions were routed to the right client methods, in order.
    assert stub.lists == [workspace]
    assert stub.reads == [workspace / "notes.txt"]
    assert stub.writes == [(workspace / "out" / "app.py", content)]
    assert stub.escape_attempts == []
    assert (workspace / "out" / "app.py").read_text(encoding="utf-8") == content

    # Each turn's result is fed back into the *next* prompt.
    assert f"Turn 1 of {MAX_TURNS}" in stub.prompts[0]
    assert f"Workspace root: {workspace}" in stub.prompts[0]
    assert "Files under .:\nnotes.txt" in stub.prompts[1]
    assert "Contents of notes.txt:\n```\noriginal note\n\n```" in stub.prompts[2]
    assert f"Wrote {len(content)} chars to out/app.py." in stub.prompts[3]
    assert f"Turn {MAX_TURNS} of {MAX_TURNS}" in stub.prompts[-1]

    assert worker.record.read == ["notes.txt"]
    assert worker.record.written == ["out/app.py"]
    assert worker.record.errors == []

    assert "Added out/app.py." in summary
    assert "Files written: out/app.py" in summary
    assert "Files read: notes.txt" in summary
    assert "turn cap" not in summary
    assert "Errors:" not in summary


def test_tool_prompt_documents_every_valid_action() -> None:
    for action in VALID_ACTIONS:
        assert f'"action": "{action}"' in WORKER_TOOL_PROMPT


# --------------------------------------------------------------------------- #
# path-escape guard (security critical)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", ESCAPE_CASES)
async def test_write_escaping_the_workspace_is_refused(tmp_path: Path, case: str) -> None:
    workspace, sentinel = _escape_layout(tmp_path)
    raw = _escape_paths(sentinel)[case]
    worker, stub = _worker(
        workspace,
        [_action("write", path=raw, content="pwned"), DONE_REPLY],
    )

    summary = await worker.run()

    # Nothing was handed to the filesystem layer at all, and the sentinel is intact.
    assert stub.writes == []
    assert stub.escape_attempts == []
    assert sentinel.read_text(encoding="utf-8") == SENTINEL_TEXT
    # Nothing new appeared anywhere under tmp_path outside the workspace.
    assert _stray_files(tmp_path, workspace) == ["outside/sentinel.txt"]

    # The refusal is surfaced to the model on the following turn.
    assert stub.complete_count == 2
    assert "ERROR" in stub.prompts[1]
    assert raw in stub.prompts[1]
    assert 'write requires a "path" inside the workspace root' in stub.prompts[1]

    assert worker.record.written == []
    assert worker.record.errors
    assert "Files written: none" in summary
    assert "Errors:" in summary
    assert "turn cap" not in summary


@pytest.mark.parametrize("case", ESCAPE_CASES)
async def test_read_escaping_the_workspace_is_refused(tmp_path: Path, case: str) -> None:
    workspace, sentinel = _escape_layout(tmp_path)
    raw = _escape_paths(sentinel)[case]
    worker, stub = _worker(workspace, [_action("read", path=raw), DONE_REPLY])

    summary = await worker.run()

    assert stub.reads == []
    assert worker.record.read == []
    assert stub.complete_count == 2
    assert "ERROR" in stub.prompts[1]
    assert raw in stub.prompts[1]
    assert 'read requires a "path" inside the workspace root' in stub.prompts[1]
    # The outside file's content must never leak into a prompt.
    assert SENTINEL_TEXT.strip() not in "".join(stub.prompts)
    assert "Errors:" in summary


def test_resolve_rejects_escaping_paths_and_accepts_inner_ones(tmp_path: Path) -> None:
    """Direct unit coverage of the guard itself, including the '.' root case."""
    workspace, sentinel = _escape_layout(tmp_path)
    (workspace / "sub").mkdir()
    (workspace / "sub" / "inner.txt").write_text("inner\n", encoding="utf-8")
    worker, _stub = _worker(workspace, [DONE_REPLY])

    resolve = worker._resolve
    for case, raw in _escape_paths(sentinel).items():
        assert resolve(raw) is None, case
    assert resolve("") is None
    assert resolve("   ") is None
    assert resolve(None) is None
    assert resolve(str(workspace / "sub" / "inner.txt")) == workspace / "sub" / "inner.txt"
    assert resolve("sub/inner.txt") == workspace / "sub" / "inner.txt"
    assert resolve("./sub/../sub/inner.txt") == workspace / "sub" / "inner.txt"
    assert resolve(".") == workspace
    assert resolve("link.txt") is None  # symlink is resolved, then rejected


async def test_relative_path_resolves_inside_workspace_and_creates_parents(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    worker, stub = _worker(
        workspace,
        [_action("write", path="sub/dir/file.txt", content="nested\n"), DONE_REPLY],
    )

    summary = await worker.run()

    target = workspace / "sub" / "dir" / "file.txt"
    assert stub.writes == [(target, "nested\n")]
    assert stub.escape_attempts == []
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == "nested\n"
    assert worker.record.written == ["sub/dir/file.txt"]
    assert "Files written: sub/dir/file.txt" in summary
    assert worker.record.errors == []


async def test_absolute_path_inside_the_workspace_is_accepted(tmp_path: Path) -> None:
    """The other half of the contract: in-workspace absolute paths are legal."""
    workspace = _workspace(tmp_path)
    target = workspace / "sub" / "abs.txt"
    worker, stub = _worker(
        workspace, [_action("write", path=str(target), content="abs\n"), DONE_REPLY]
    )

    summary = await worker.run()

    assert stub.writes == [(target, "abs\n")]
    assert target.read_text(encoding="utf-8") == "abs\n"
    assert worker.record.written == ["sub/abs.txt"]
    assert worker.record.errors == []
    assert "turn cap" not in summary


# --------------------------------------------------------------------------- #
# malformed replies: reported, never raised
# --------------------------------------------------------------------------- #

MALFORMED_REPLIES: list[tuple[str, str]] = [
    pytest.param("", "empty reply", id="empty"),
    pytest.param("   \n  ", "empty reply", id="whitespace-only"),
    pytest.param(
        "Sure! Let me think about the workspace first.",
        "reply was not JSON",
        id="prose-only",
    ),
    pytest.param('{"action":"read",}', "invalid JSON", id="invalid-json"),
    pytest.param('{"path": "a.txt", "content": "x"}', 'missing "action" key', id="no-action"),
    pytest.param(
        '{"action": 7, "path": "a.txt"}', 'missing "action" key', id="non-string-action"
    ),
    pytest.param(
        '{"action": "delete", "path": "a.txt"}',
        'unknown action "delete"',
        id="unknown-action",
    ),
    pytest.param(
        '{"action": "read"}', 'read requires a "path"', id="read-without-path"
    ),
    pytest.param(
        '{"action": "write", "content": "x"}',
        'write requires a "path"',
        id="write-without-path",
    ),
    pytest.param(
        '{"action": "read", "path": 7}', '"path" must be a string', id="non-string-path"
    ),
    pytest.param(
        '{"action": "write", "path": "a.txt"}',
        'write to a.txt is missing a "content" field',
        id="write-without-content",
    ),
]


@pytest.mark.parametrize(("reply", "expected"), MALFORMED_REPLIES)
async def test_malformed_reply_is_reported_and_the_loop_continues(
    tmp_path: Path, reply: str, expected: str
) -> None:
    workspace = _workspace(tmp_path)
    worker, stub = _worker(workspace, [reply, DONE_REPLY])

    summary = await worker.run()  # must not raise

    assert stub.complete_count == 2
    assert stub.models == [WORKER_MODEL] * 2
    assert "ERROR" in stub.prompts[1]
    assert expected in stub.prompts[1]
    # A malformed action is never executed.
    assert stub.writes == []
    assert stub.reads == []
    assert stub.lists == []
    # The loop recovered: the scripted 'done' reply on turn 2 ended the run.
    assert "finished the task" in summary
    assert "turn cap" not in summary


async def test_read_of_a_missing_file_is_reported_instead_of_crashing(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    worker, stub = _worker(workspace, [_action("read", path="nope.txt"), DONE_REPLY])

    summary = await worker.run()

    assert stub.reads == [workspace / "nope.txt"]
    assert "ERROR" in stub.prompts[1]
    assert "could not read nope.txt" in stub.prompts[1]
    assert "No such file or directory" in stub.prompts[1]
    # The read never reached the record, but the failure did.
    assert worker.record.read == []
    assert worker.record.errors
    assert "Errors:" in summary
    assert "turn cap" not in summary


async def test_reading_or_writing_a_directory_is_refused(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "sub").mkdir()
    worker, stub = _worker(
        workspace,
        [
            _action("read", path="sub"),
            _action("write", path=".", content="nope"),
            DONE_REPLY,
        ],
    )

    summary = await worker.run()

    assert stub.complete_count == 3
    assert stub.writes == []
    assert 'sub is a directory; use action "list" instead.' in stub.prompts[1]
    assert "is a directory; writes are only allowed to files." in stub.prompts[2]
    assert len(worker.record.errors) == 2
    assert "Errors:" in summary


async def test_parse_errors_are_recorded_in_the_final_summary(tmp_path: Path) -> None:
    """A cycle whose every reply was unparsable reports those failures, not a clean run.

    ``run`` routes an unparsable reply through ``_fail``, so each parse failure is
    appended to ``record.errors`` and reaches the summary's ``Errors:`` line. The
    complementary direction is pinned by ``test_happy_path_list_read_write_done``:
    a run with no malformed replies records nothing and omits the section entirely.
    """
    workspace = _workspace(tmp_path)
    worker, stub = _worker(
        workspace,
        ["total nonsense", "still nonsense", "no json here", DONE_REPLY],
    )

    summary = await worker.run()

    # The loop still runs exactly as many turns as before and recovers on turn 4.
    assert stub.complete_count == MAX_TURNS
    # The text sent to the model is unchanged: every later prompt still carries the
    # ERROR feedback, and the parse-error fragment still reaches the final prompt.
    for prompt in stub.prompts[1:]:
        assert "ERROR" in prompt
    assert "reply was not JSON" in stub.prompts[-1]
    # One recorded error per malformed reply, in order.
    assert worker.record.errors == [
        "reply was not JSON: 'total nonsense'",
        "reply was not JSON: 'still nonsense'",
        "reply was not JSON: 'no json here'",
    ]
    # And the summary lists them instead of claiming a clean result.
    assert (
        "Errors: reply was not JSON: 'total nonsense'; "
        "reply was not JSON: 'still nonsense'; "
        "reply was not JSON: 'no json here'"
    ) in summary
    assert "Files written: none" in summary
    assert "finished the task" in summary


# --------------------------------------------------------------------------- #
# turn cap
# --------------------------------------------------------------------------- #


async def test_loop_stops_at_the_turn_cap(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "a.txt").write_text("a\n", encoding="utf-8")
    # A valid, repeatable, non-done action: the loop can only end via the cap.
    worker, stub = _worker(workspace, [_action("list", path=".")], repeat_last=True)

    summary = await worker.run()

    assert stub.complete_count == MAX_TURNS
    assert stub.models == [WORKER_MODEL] * MAX_TURNS
    for turn in range(1, MAX_TURNS + 1):
        assert f"Turn {turn} of {MAX_TURNS}" in stub.prompts[turn - 1]
    assert stub.lists == [workspace] * MAX_TURNS
    assert f"Worker stopped at the turn cap ({MAX_TURNS}/{MAX_TURNS})" in summary
    assert "turn cap" in summary
    assert "Worker reported done without a summary." not in summary


async def test_done_without_a_summary_still_reports_files_touched(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "old.txt").write_text("old\n", encoding="utf-8")
    worker, stub = _worker(
        workspace,
        [
            _action("read", path="old.txt"),
            _action("write", path="new.txt", content="new\n"),
            json.dumps({"action": "done"}),
        ],
    )

    summary = await worker.run()

    assert stub.complete_count == 3
    assert "Worker reported done without a summary." in summary
    assert "Files written: new.txt" in summary
    assert "Files read: old.txt" in summary
    assert "turn cap" not in summary
    assert worker.record.errors == []


# --------------------------------------------------------------------------- #
# list feedback and truncation
# --------------------------------------------------------------------------- #


async def test_list_output_is_fed_back_to_the_model(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "pkg").mkdir()
    (workspace / "pkg" / "b.py").write_text("b\n", encoding="utf-8")
    (workspace / "a.py").write_text("a\n", encoding="utf-8")
    worker, stub = _worker(workspace, [_action("list", path="."), DONE_REPLY])

    await worker.run()

    assert stub.lists == [workspace]
    assert "Files under .:\na.py\npkg/b.py" in stub.prompts[1]
    assert worker.record.errors == []


async def test_list_of_an_empty_or_missing_directory_is_not_an_error(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    worker, stub = _worker(workspace, [_action("list", path="empty"), DONE_REPLY])

    summary = await worker.run()

    assert "Files under empty:\n(no files)" in stub.prompts[1]
    assert worker.record.errors == []
    assert "Errors:" not in summary


async def test_list_output_is_truncated_at_max_list_entries(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    many = workspace / "many"
    many.mkdir()
    for index in range(MAX_LIST_ENTRIES + 1):
        (many / f"f{index:03d}.txt").write_text("x\n", encoding="utf-8")
    worker, stub = _worker(workspace, [_action("list", path="many"), DONE_REPLY])

    await worker.run()

    assert stub.lists == [many]
    body = stub.prompts[1]
    assert "Files under many:\nf000.txt" in body
    assert f"f{MAX_LIST_ENTRIES - 1:03d}.txt" in body
    # The one entry past the cap is dropped and counted, not silently lost.
    assert f"f{MAX_LIST_ENTRIES:03d}.txt" not in body
    assert "[1 more files not shown]" in body
    assert body.count(".txt") == MAX_LIST_ENTRIES
    assert worker.record.errors == []


async def test_long_reads_are_truncated_at_max_read_chars(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    long_text = "x" * (MAX_READ_CHARS + 500)
    (workspace / "big.txt").write_text(long_text, encoding="utf-8")
    worker, stub = _worker(workspace, [_action("read", path="big.txt"), DONE_REPLY])

    summary = await worker.run()

    assert worker.record.read == ["big.txt"]
    assert worker.record.errors == []
    assert long_text not in stub.prompts[1]
    assert "x" * MAX_READ_CHARS in stub.prompts[1]
    assert f"... [truncated at {MAX_READ_CHARS} chars]" in stub.prompts[1]
    assert "Files read: big.txt" in summary


# --------------------------------------------------------------------------- #
# documented deviations from the tool protocol
# --------------------------------------------------------------------------- #


async def test_json_array_reply_is_unwrapped_and_executed(tmp_path: Path) -> None:
    """DEVIATION: a JSON *array* of actions is silently unwrapped and executed.

    ``_parse`` slices ``text[text.find("{"):text.rfind("}") + 1]``, so a reply such
    as ``[{"action": "write", ...}]`` is reduced to its single object and run, even
    though the protocol demands "exactly ONE JSON action object per turn". The
    ``isinstance(payload, dict)`` rejection branch is therefore unreachable, because
    any slice that starts with ``{`` and ends with ``}`` and parses as JSON is
    necessarily an object. Pinned as-is: only a single-element array escapes here,
    and a multi-element one is silently truncated to its first object.
    """
    workspace = _workspace(tmp_path)
    reply = json.dumps([{"action": "write", "path": "a.txt", "content": "hi"}])
    worker, stub = _worker(workspace, [reply, DONE_REPLY])

    summary = await worker.run()

    assert stub.writes == [(workspace / "a.txt", "hi")]
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "hi"
    assert worker.record.written == ["a.txt"]
    assert worker.record.errors == []
    assert "Files written: a.txt" in summary


async def test_non_string_content_is_json_encoded_into_the_file(tmp_path: Path) -> None:
    """DEVIATION: a non-string ``content`` is coerced with ``json.dumps``, not rejected.

    ``{"content": 123}`` writes the literal text ``123`` instead of reporting an
    error, so a model that sends a number or a nested object gets a surprising file
    rather than a correction prompt. Pinned as-is.
    """
    workspace = _workspace(tmp_path)
    worker, stub = _worker(
        workspace,
        [_action("write", path="num.txt", content=123), DONE_REPLY],
    )

    summary = await worker.run()

    assert stub.writes == [(workspace / "num.txt", "123")]
    assert (workspace / "num.txt").read_text(encoding="utf-8") == "123"
    assert worker.record.errors == []
    assert "Errors:" not in summary


# --------------------------------------------------------------------------- #
# protocol tolerance and state
# --------------------------------------------------------------------------- #


async def test_action_names_are_trimmed_and_case_insensitive(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    reply = json.dumps({"action": "  WRITE  ", "path": "a.txt", "content": "hi\n"})
    worker, stub = _worker(workspace, [reply, DONE_REPLY])

    summary = await worker.run()

    assert stub.writes == [(workspace / "a.txt", "hi\n")]
    assert worker.record.written == ["a.txt"]
    assert worker.record.errors == []
    assert "turn cap" not in summary


async def test_prose_and_code_fences_around_the_action_are_tolerated(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    reply = "Here you go:\n```json\n" + _action("write", path="a.txt", content="hi\n") + "\n```"
    worker, stub = _worker(workspace, [reply, DONE_REPLY])

    await worker.run()

    assert stub.writes == [(workspace / "a.txt", "hi\n")]
    assert worker.record.errors == []


async def test_record_is_reset_when_run_is_called_again(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    worker, stub = _worker(
        workspace,
        [
            _action("write", path="first.txt", content="1\n"),
            DONE_REPLY,
            json.dumps({"action": "done", "summary": "second run"}),
        ],
    )

    first = await worker.run()
    second = await worker.run()

    assert stub.complete_count == 3
    assert "Files written: first.txt" in first
    assert "Files read: none" in first
    assert "Files written: none" in second
    assert "Files read: none" in second
    assert worker.record.written == []
    assert worker.record.read == []


async def test_worker_uses_the_client_it_is_given(tmp_path: Path) -> None:
    """A client object must not be swapped out for a real LLMClient."""
    workspace = _workspace(tmp_path)
    worker, stub = _worker(workspace, [DONE_REPLY])

    assert worker.client is stub
    assert worker.workspace == workspace

    summary = await worker.run()

    assert stub.complete_count == 1
    assert "Task task-1: implement the thing" in summary
    assert stub.escape_attempts == []
