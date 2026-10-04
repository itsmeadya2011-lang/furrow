"""Unit tests for the agent layer: the tool sandbox, and the three agents.

`furrow/agents/tools.py` had no direct unit coverage at all - its sandbox was only ever
exercised end-to-end by `test_orchestrator.py`, where a containment regression would be
invisible. This module tests `ToolRegistry` directly, with `tmp_path` as the sandbox root,
and treats the security properties (containment, atomic writes, never-raise dispatch) as
first-class assertions rather than smoke tests.

Everything here is offline: no provider client is ever built, and every subprocess runs
with its working directory inside the sandbox or inside pytest's basetemp. Nothing is
written inside the repository.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from furrow.agents import tester as tester_agent
from furrow.agents.planner import PlannerAgent
from furrow.agents.prompts import PLANNER_PROMPT, TESTER_PROMPT, WORKER_SYSTEM_PROMPT
from furrow.agents.tools import (
    MAX_LIST_RESULTS,
    MAX_OUTPUT_CHARS,
    PREVIEW_CHARS,
    PREVIEW_SAFE_ARGS,
    ToolContext,
    ToolRegistry,
)
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TaskStatus, TestResult, reload_settings
from furrow.events import Event, EventBus, EventType
from furrow.llm import (
    LLMClient,
    LLMConfigurationError,
    LLMResponseError,
    ToolCall,
    ToolResult,
)
from tests.conftest import DEFAULT_TEST, ScriptedLLMClient, as_llm_client

GOAL = "make the thing work"
SECRET = "TOP-SECRET-VALUE"
SECRET_BODY = f"{SECRET}\n"


# --------------------------------------------------------------------- helpers
def build_settings(workspace: Path, **overrides: Any) -> Settings:
    """Settings with the same bounds as the shared `settings` fixture, plus overrides."""
    defaults: dict[str, Any] = {
        "workspace": workspace,
        "max_cycles": 2,
        "max_stalled_cycles": 2,
        "test_timeout": 60,
        "state_file": None,
    }
    return reload_settings(**{**defaults, **overrides})


def make_registry(root: Path, settings: Settings, bus: EventBus | None = None) -> ToolRegistry:
    """A registry rooted at `root`, the only directory any tool may touch."""
    return ToolRegistry(ToolContext(workspace=root, settings=settings, bus=bus))


def command(script: str) -> str:
    """A shell-free command string; the tester re-splits it with shlex."""
    return shlex.join([sys.executable, "-c", script])


PASSING_COMMAND = command("import sys; sys.exit(0)")
FAILING_COMMAND = command("import sys; sys.exit(1)")
NOISE_CHARACTERS = 200_000
NOISE_COMMAND = command(
    "import sys; sys.stdout.write('x' * 200000); sys.stdout.write('TAIL-MARKER\\n')"
)
SLEEP_COMMAND = f"exec {shlex.join([sys.executable, '-c', 'import time; time.sleep(30)'])}"


def events_of(bus: EventBus, kind: EventType) -> list[Event]:
    return [event for event in bus.history() if event.type is kind]


def make_task(task_id: str = "1", **overrides: Any) -> TaskModel:
    return TaskModel(
        id=task_id,
        description=overrides.pop("description", "write greeting.py"),
        files=overrides.pop("files", ["greeting.py"]),
        **overrides,
    )


class _ErrorStopClient(ScriptedLLMClient):
    """A scripted client whose tool rounds always end with `stop_reason="error"`."""

    async def complete_with_tools(
        self,
        prompt: str,
        tools: Sequence[Any],
        *,
        system: str | None = None,
        model: str | None = None,
        max_iterations: int | None = None,
        on_tool_call: Callable[[ToolCall], Awaitable[None]] | None = None,
    ) -> ToolResult:
        self._record(
            "complete_with_tools",
            prompt,
            system,
            model,
            tools=[tool["name"] for tool in tools],
            max_iterations=max_iterations,
        )
        return ToolResult(
            text="[error] RateLimitError: too many requests",
            tool_calls=[],
            iterations=1,
            stop_reason="error",
        )


class _LeakyDecoderClient(ScriptedLLMClient):
    """A client that leaks a bare `json.JSONDecodeError` out of `complete_json`."""

    async def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_attempts: int = 2,
    ) -> Any:
        self._record("complete_json", prompt, system, model, max_attempts=max_attempts)
        raise json.JSONDecodeError("Expecting value: line 1 column 1 (char 0)", "", 0)


class _NoToolClient:
    """A client with no `complete_with_tools` attribute at all."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def complete(self, prompt: str, **_kwargs: Any) -> str:
        return ""


@pytest.fixture
def outside(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory outside the sandbox holding the file no tool may ever reach."""
    path = tmp_path_factory.mktemp("outside_sandbox")
    (path / "secret.txt").write_text(SECRET_BODY, encoding="utf-8")
    return path


@pytest.fixture
def escape_workspace(workspace: Path, outside: Path) -> Path:
    """The sandbox root, decorated with links that point out of it."""
    (workspace / "nested").mkdir()
    (workspace / "link.txt").symlink_to(outside / "secret.txt")
    (workspace / "outdir").symlink_to(outside, target_is_directory=True)
    return workspace


# ------------------------------------------------------- registry construction
def test_registry_exposes_the_resolved_root(workspace: Path, settings: Settings) -> None:
    assert make_registry(workspace, settings).workspace == workspace.resolve()


def test_registry_raises_when_the_workspace_is_a_file(workspace: Path) -> None:
    blocker = workspace / "not-a-directory.txt"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="is not an existing directory"):
        make_registry(blocker, build_settings(blocker))


def test_specs_are_json_tool_definitions(workspace: Path, settings: Settings) -> None:
    specs = make_registry(workspace, settings).specs()
    assert [spec["name"] for spec in specs] == [
        "read_file",
        "write_file",
        "edit_file",
        "list_files",
        "search",
        "run_command",
    ]
    assert all(spec["parameters"]["type"] == "object" for spec in specs)
    assert all(isinstance(spec["description"], str) and spec["description"] for spec in specs)


def test_specs_are_a_fresh_deep_copy(workspace: Path, settings: Settings) -> None:
    registry = make_registry(workspace, settings)
    first = registry.specs()
    first[0]["description"] = "mutated"
    assert registry.specs()[0]["description"] != "mutated"


# ------------------------------------------------------------------- read_file
async def test_read_file_returns_the_contents(workspace: Path, settings: Settings) -> None:
    (workspace / "notes.md").write_text("hello sandbox\n", encoding="utf-8")
    result = await make_registry(workspace, settings).invoke("read_file", {"path": "notes.md"})
    assert result == "hello sandbox\n"


async def test_read_file_refuses_a_missing_file(workspace: Path, settings: Settings) -> None:
    result = await make_registry(workspace, settings).invoke("read_file", {"path": "nope.md"})
    assert result.startswith("ERROR:")
    assert "nope.md does not exist" in result


async def test_read_file_refuses_a_directory(workspace: Path, settings: Settings) -> None:
    (workspace / "pkg").mkdir()
    result = await make_registry(workspace, settings).invoke("read_file", {"path": "pkg"})
    assert result.startswith("ERROR:")
    assert "is a directory, not a file" in result
    assert "list_files" in result


async def test_read_file_refuses_binary_content(workspace: Path, settings: Settings) -> None:
    (workspace / "blob.bin").write_bytes(b"\x00\x01\x02binary")
    result = await make_registry(workspace, settings).invoke("read_file", {"path": "blob.bin"})
    assert result.startswith("ERROR:")
    assert "binary" in result
    assert "NUL byte" in result


async def test_read_file_truncates_at_max_bytes_with_a_marker(
    workspace: Path, settings: Settings
) -> None:
    (workspace / "big.txt").write_text("a" * 1_000, encoding="utf-8")
    registry = make_registry(workspace, settings)
    result = await registry.invoke("read_file", {"path": "big.txt", "max_bytes": 10})
    assert result == "a" * 10 + "\n... [truncated at 10 of 1000 bytes]"


async def test_read_file_clamps_a_zero_max_bytes_to_one(
    workspace: Path, settings: Settings
) -> None:
    (workspace / "big.txt").write_text("a" * 40, encoding="utf-8")
    result = await make_registry(workspace, settings).invoke(
        "read_file", {"path": "big.txt", "max_bytes": 0}
    )
    assert result == "a\n... [truncated at 1 of 40 bytes]"


async def test_read_file_caps_the_output_size(workspace: Path, settings: Settings) -> None:
    (workspace / "huge.txt").write_text("b" * 120_000, encoding="utf-8")
    result = await make_registry(workspace, settings).invoke("read_file", {"path": "huge.txt"})
    assert len(result) <= MAX_OUTPUT_CHARS
    assert "truncated" in result


# ------------------------------------------------------------------ write_file
async def test_write_file_creates_parent_directories(workspace: Path, settings: Settings) -> None:
    result = await make_registry(workspace, settings).invoke(
        "write_file", {"path": "src/deep/mod.py", "content": "x = 1\n"}
    )
    assert "src/deep/mod.py" in result
    assert (workspace / "src" / "deep" / "mod.py").read_text(encoding="utf-8") == "x = 1\n"


async def test_write_file_records_the_workspace_relative_path(
    workspace: Path, settings: Settings
) -> None:
    registry = make_registry(workspace, settings)
    await registry.invoke("write_file", {"path": "src/a.py", "content": "a"})
    await registry.invoke("write_file", {"path": "src/a.py", "content": "aa"})
    await registry.invoke("write_file", {"path": "b.txt", "content": "b"})
    assert registry.changed_files == ["src/a.py", "b.txt"]


async def test_write_file_is_atomic_and_leaves_no_temp_files(
    workspace: Path, settings: Settings
) -> None:
    target = workspace / "pkg" / "mod.py"
    registry = make_registry(workspace, settings)
    await registry.invoke("write_file", {"path": "pkg/mod.py", "content": "clean\n"})
    assert [entry.name for entry in (workspace / "pkg").iterdir()] == ["mod.py"]
    assert target.read_text(encoding="utf-8") == "clean\n"


async def test_write_file_overwrites_an_existing_file(workspace: Path, settings: Settings) -> None:
    (workspace / "mod.py").write_text("old\n", encoding="utf-8")
    await make_registry(workspace, settings).invoke(
        "write_file", {"path": "mod.py", "content": "new\n"}
    )
    assert (workspace / "mod.py").read_text(encoding="utf-8") == "new\n"


async def test_write_file_accepts_empty_content(workspace: Path, settings: Settings) -> None:
    result = await make_registry(workspace, settings).invoke(
        "write_file", {"path": "empty.txt", "content": ""}
    )
    assert "Wrote 0 bytes" in result
    assert (workspace / "empty.txt").read_text(encoding="utf-8") == ""


async def test_write_file_leaves_no_temp_behind_when_the_target_is_a_directory(
    workspace: Path, settings: Settings
) -> None:
    (workspace / "pkg").mkdir()
    (workspace / "pkg" / "keep.txt").write_text("keep", encoding="utf-8")
    result = await make_registry(workspace, settings).invoke(
        "write_file", {"path": "pkg", "content": "x"}
    )
    assert result.startswith("ERROR:")
    assert sorted(p.name for p in (workspace / "pkg").iterdir()) == ["keep.txt"]


# ------------------------------------------------------------------- edit_file
async def test_edit_file_replaces_a_unique_occurrence(workspace: Path, settings: Settings) -> None:
    (workspace / "mod.py").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    result = await make_registry(workspace, settings).invoke(
        "edit_file", {"path": "mod.py", "old_string": "beta", "new_string": "BETA"}
    )
    assert "Made 1 replacement(s)" in result
    assert (workspace / "mod.py").read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"


async def test_edit_file_fails_when_old_string_is_absent(
    workspace: Path, settings: Settings
) -> None:
    (workspace / "mod.py").write_text("alpha\n", encoding="utf-8")
    result = await make_registry(workspace, settings).invoke(
        "edit_file", {"path": "mod.py", "old_string": "omega", "new_string": "x"}
    )
    assert result.startswith("ERROR:")
    assert "old_string was not found in mod.py" in result
    assert (workspace / "mod.py").read_text(encoding="utf-8") == "alpha\n"


async def test_edit_file_refuses_an_ambiguous_match(workspace: Path, settings: Settings) -> None:
    (workspace / "mod.py").write_text("dup\nother\ndup\n", encoding="utf-8")
    result = await make_registry(workspace, settings).invoke(
        "edit_file", {"path": "mod.py", "old_string": "dup", "new_string": "unique"}
    )
    assert result.startswith("ERROR:")
    assert "ambiguous" in result
    assert "appears 2 times" in result
    assert (workspace / "mod.py").read_text(encoding="utf-8") == "dup\nother\ndup\n"


async def test_edit_file_replaces_all_when_replace_all_is_true(
    workspace: Path, settings: Settings
) -> None:
    (workspace / "mod.py").write_text("dup\nother\ndup\n", encoding="utf-8")
    result = await make_registry(workspace, settings).invoke(
        "edit_file",
        {"path": "mod.py", "old_string": "dup", "new_string": "same", "replace_all": True},
    )
    assert "Made 2 replacement(s)" in result
    assert (workspace / "mod.py").read_text(encoding="utf-8") == "same\nother\nsame\n"


async def test_edit_file_refuses_an_empty_old_string(workspace: Path, settings: Settings) -> None:
    (workspace / "mod.py").write_text("alpha\n", encoding="utf-8")
    result = await make_registry(workspace, settings).invoke(
        "edit_file", {"path": "mod.py", "old_string": "", "new_string": "x"}
    )
    assert result.startswith("ERROR:")
    assert "old_string must not be empty" in result


async def test_edit_file_records_the_change(workspace: Path, settings: Settings) -> None:
    (workspace / "mod.py").write_text("alpha\n", encoding="utf-8")
    registry = make_registry(workspace, settings)
    await registry.invoke(
        "edit_file", {"path": "mod.py", "old_string": "alpha", "new_string": "beta"}
    )
    assert registry.changed_files == ["mod.py"]


# ------------------------------------------------------------------ list_files
async def test_list_files_honours_exclude_dirs(workspace: Path, settings: Settings) -> None:
    for excluded in (".git", "__pycache__", "demo.egg-info", "node_modules"):
        directory = workspace / excluded
        directory.mkdir()
        (directory / "hidden.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    result = await make_registry(workspace, settings).invoke("list_files", {"path": "."})
    assert result == "src/app.py"


async def test_list_files_honours_a_pattern(workspace: Path, settings: Settings) -> None:
    (workspace / "a.py").write_text("", encoding="utf-8")
    (workspace / "b.txt").write_text("", encoding="utf-8")
    (workspace / "pkg").mkdir()
    (workspace / "pkg" / "c.py").write_text("", encoding="utf-8")
    result = await make_registry(workspace, settings).invoke("list_files", {"pattern": "*.py"})
    assert result.splitlines() == ["a.py", "pkg/c.py"]


async def test_list_files_truncates_at_the_cap_with_an_explicit_marker(
    workspace: Path, settings: Settings
) -> None:
    data = workspace / "data"
    data.mkdir()
    for index in range(MAX_LIST_RESULTS + 5):
        (data / f"f{index:03d}.txt").write_text("", encoding="utf-8")
    result = await make_registry(workspace, settings).invoke("list_files", {"path": "."})
    lines = result.splitlines()
    assert len(lines) == MAX_LIST_RESULTS + 1
    assert f"{MAX_LIST_RESULTS + 5} files match" in lines[-1]
    assert f"truncated to the first {MAX_LIST_RESULTS}" in lines[-1]
    assert f"data/f{MAX_LIST_RESULTS + 4:03d}.txt" not in lines


async def test_list_files_reports_an_empty_result(workspace: Path, settings: Settings) -> None:
    result = await make_registry(workspace, settings).invoke("list_files", {"path": "."})
    assert "No files under . match" in result


async def test_list_files_refuses_a_file_as_the_root(workspace: Path, settings: Settings) -> None:
    (workspace / "a.txt").write_text("", encoding="utf-8")
    result = await make_registry(workspace, settings).invoke("list_files", {"path": "a.txt"})
    assert result.startswith("ERROR:")
    assert "is not an existing directory" in result


async def test_list_files_emits_a_json_safe_tool_call_event(
    workspace: Path, settings: Settings, bus: EventBus
) -> None:
    (workspace / "a.txt").write_text("x" * 50_000, encoding="utf-8")
    await make_registry(workspace, settings, bus).invoke("list_files", {"path": "."})
    events = events_of(bus, EventType.TOOL_CALL)
    assert len(events) == 1
    assert events[0].data["tool"] == "list_files"
    assert events[0].data["ok"] is True
    assert json.loads(json.dumps(events[0].to_dict()))


async def test_tool_call_event_message_never_carries_file_content(
    workspace: Path, settings: Settings, bus: EventBus
) -> None:
    """SECURITY: a `tool_call` event must not carry file contents anywhere.

    This test used to invoke only `read_file`, whose sole argument is `path`, so it could
    not fail: the arguments that actually carry content are `write_file.content` and
    `edit_file.old_string`/`new_string`, and those were rendered into the event MESSAGE
    (the `data` payload was always clean). Each tool now gets its own unique marker, so a
    leak out of any one of them fails here rather than reaching the event ring buffer, the
    `/ws` stream or `GET /api/runs/{id}`.

    WIDENED (residual-leak pass). Two routes were still unproven here:

    * `edit_file.old_string` was pinned only by the exact-message assertion below, which is
      legitimate but blunt - a message that happened to omit the whole `edit_file` call
      would also pass. It now carries its own marker like `new_string`.
    * `search.query` was NOT covered at all, and it was leaking. `search` is a LITERAL
      substring search, so its `query` is text copied verbatim out of a file the workspace
      holds. The marker below is planted in `app.py` and then searched for verbatim, which
      is exactly how a model hunts for a credential it has seen.
    """
    read_marker = "ZZ-READ-CONTENT-MARKER-ZZ"
    write_marker = "ZZ-WRITE-CONTENT-MARKER-ZZ"
    old_marker = "ZZ-EDIT-OLD-CONTENT-MARKER-ZZ"
    edit_marker = "ZZ-EDIT-CONTENT-MARKER-ZZ"
    search_marker = "ZZ-SEARCH-QUERY-CONTENT-MARKER-ZZ"
    markers = (read_marker, write_marker, old_marker, edit_marker, search_marker)
    (workspace / "app.py").write_text(
        f"# {read_marker}\nold = {old_marker}\nsecret = {search_marker}\n", encoding="utf-8"
    )

    registry = make_registry(workspace, settings, bus)
    await registry.invoke("read_file", {"path": "app.py"})
    await registry.invoke(
        "write_file", {"path": "written.py", "content": f"y = 2  # {write_marker}\n"}
    )
    await registry.invoke(
        "edit_file",
        {
            "path": "app.py",
            "old_string": f"old = {old_marker}",
            "new_string": f"old = 3  # {edit_marker}",
        },
    )
    await registry.invoke("search", {"query": search_marker, "path": "."})

    emitted = events_of(bus, EventType.TOOL_CALL)
    assert [event.data["tool"] for event in emitted] == [
        "read_file",
        "write_file",
        "edit_file",
        "search",
    ]
    for event in emitted:
        payload = json.dumps(event.to_dict())
        data = json.dumps(event.data)
        for marker in markers:
            assert marker not in event.message, event.message
            assert marker not in data, data
            assert marker not in payload, payload

    # The search actually matched, so the tool ran and its detail payload is live: the
    # guarantee is that the COUNT is available and the query text is not.
    assert emitted[3].data["matches"] == 1
    assert emitted[3].data["path"] == "."

    # The message stays readable: identifiers still render, content never does.
    assert emitted[0].message == "read_file(path=app.py)"
    assert emitted[1].message == "write_file(path=written.py)"
    assert emitted[2].message == "edit_file(path=app.py)"
    assert emitted[3].message == "search(path=.)"


async def test_tool_call_event_never_carries_content_on_the_error_path(
    workspace: Path, settings: Settings, bus: EventBus
) -> None:
    """SECURITY: the drop is unconditional, not something that only holds when a tool works.

    The `ok=False` branch is the one an attacker-shaped call lands on: a bad `old_string`
    fails the unique-match check, and the message is the only place the argument could have
    leaked. Both the "not found" and the "ambiguous" error text quote the model back, so
    this pins the error paths too.
    """
    missing_marker = "ZZ-EDIT-OLD-MISSING-MARKER-ZZ"
    ambiguous_marker = "ZZ-EDIT-OLD-AMBIGUOUS-MARKER-ZZ"
    (workspace / "dup.txt").write_text(f"x\n{ambiguous_marker}\nx\n{ambiguous_marker}\n", "utf-8")

    registry = make_registry(workspace, settings, bus)
    await registry.invoke(
        "edit_file",
        {
            "path": "app.py",
            "old_string": missing_marker,
            "new_string": "irrelevant",
        },
    )
    await registry.invoke(
        "write_file", {"path": "out.txt", "content": f"{ambiguous_marker}\n"}
    )
    await registry.invoke(
        "edit_file",
        {
            "path": "dup.txt",
            "old_string": ambiguous_marker,
            "new_string": "irrelevant",
        },
    )

    emitted = events_of(bus, EventType.TOOL_CALL)
    assert [event.data["ok"] for event in emitted] == [False, True, False]
    for event in emitted:
        payload = json.dumps(event.to_dict())
        for marker in (missing_marker, ambiguous_marker):
            assert marker not in event.message, event.message
            assert marker not in payload, payload
    assert emitted[0].message == "edit_file(path=app.py)"
    assert emitted[2].message == "edit_file(path=dup.txt)"


async def test_the_assembled_tool_call_message_is_bounded_by_preview_chars(
    workspace: Path, settings: Settings, bus: EventBus
) -> None:
    """SECURITY/ROBUSTNESS: `PREVIEW_CHARS` bounds the MESSAGE, tool name included.

    `PREVIEW_CHARS` used to cap only the argument LIST, which `_emit` then wrapped as
    `f"{name}({preview})"`. The real ceiling was therefore `PREVIEW_CHARS + len(name) + 2`
    (253 for `run_command`), so the constant did not mean what a reader of it would assume.
    The cap is now applied to the assembled string, and this pins that.

    The renderer walks `PREVIEW_SAFE_ARGS` by NAME over whatever keys the caller sent, and
    arguments are untrusted JSON from a model, so the case that exercises the ceiling is a
    call that populates every allowlisted key. No single tool call through the published
    specs gets close - each value is clipped to `SNIPPET_CHARS` first and the specs use at
    most three keys - which is precisely why the cap has to be an invariant rather than
    something reachable through the schemas.
    """
    filler = "".join(chr(97 + index % 26) for index in range(5_000))
    arguments: dict[str, Any] = {key: filler for key in PREVIEW_SAFE_ARGS if key != "command"}
    arguments["command"] = command("pass")

    registry = make_registry(workspace, settings, bus)
    await registry.invoke("run_command", arguments)
    # A realistic call, to keep the readability of a short message pinned too.
    await registry.invoke("run_command", {"command": command("pass"), "timeout": 30})

    emitted = events_of(bus, EventType.TOOL_CALL)
    assert [event.data["tool"] for event in emitted] == ["run_command", "run_command"]

    # The old worst case, for reference: 240 + len("run_command") + 2 == 253.
    assert PREVIEW_CHARS + len("run_command") + 2 == 253

    crowded = emitted[0].message
    # Clipping actually happened, so this is not passing vacuously on a short message.
    assert len(crowded) == PREVIEW_CHARS
    assert len(crowded) <= PREVIEW_CHARS
    # The tool name survives at the head, which is what makes the message readable.
    assert crowded.startswith("run_command(path=")
    assert crowded.endswith(")")

    # The realistic call renders in full and stays legible.
    short = emitted[1].message
    assert short == f"run_command(command={arguments['command']}, timeout=30)"
    assert len(short) < PREVIEW_CHARS


# ---------------------------------------------------------------------- search
async def test_search_returns_path_line_text_format(workspace: Path, settings: Settings) -> None:
    (workspace / "a.txt").write_text("first line\nneedle here\n", encoding="utf-8")
    result = await make_registry(workspace, settings).invoke("search", {"query": "needle"})
    assert result == "a.txt:2: needle here"


async def test_search_skips_excluded_dirs_and_binary_files(
    workspace: Path, settings: Settings
) -> None:
    hidden = workspace / ".git"
    hidden.mkdir()
    (hidden / "config").write_text("needle in git\n", encoding="utf-8")
    (workspace / "blob.dat").write_bytes(b"needle\x00binary")
    (workspace / "src").mkdir()
    (workspace / "src" / "keep.txt").write_text("needle kept\n", encoding="utf-8")
    result = await make_registry(workspace, settings).invoke("search", {"query": "needle"})
    assert result == "src/keep.txt:1: needle kept"


async def test_search_respects_max_results(workspace: Path, settings: Settings) -> None:
    (workspace / "many.txt").write_text("needle\n" * 10, encoding="utf-8")
    result = await make_registry(workspace, settings).invoke(
        "search", {"query": "needle", "max_results": 3}
    )
    lines = result.splitlines()
    assert len(lines) == 4
    assert lines[-1] == "... [stopped after 3 matches; narrow the query or the path]"


async def test_search_reports_no_matches(workspace: Path, settings: Settings) -> None:
    (workspace / "a.txt").write_text("alpha\n", encoding="utf-8")
    result = await make_registry(workspace, settings).invoke("search", {"query": "omega"})
    assert result == "No matches for 'omega' under .."


async def test_search_refuses_an_empty_query(workspace: Path, settings: Settings) -> None:
    result = await make_registry(workspace, settings).invoke("search", {"query": ""})
    assert result.startswith("ERROR:")
    assert "query must not be empty" in result


async def test_search_refuses_a_file_as_the_root(workspace: Path, settings: Settings) -> None:
    (workspace / "a.txt").write_text("needle\n", encoding="utf-8")
    result = await make_registry(workspace, settings).invoke(
        "search", {"query": "needle", "path": "a.txt"}
    )
    assert result.startswith("ERROR:")
    assert "is not an existing directory" in result


# ----------------------------------------------------------------- run_command
async def test_run_command_reports_a_zero_exit_code(workspace: Path, settings: Settings) -> None:
    result = await make_registry(workspace, settings).invoke("run_command", {"command": "true"})
    assert result == "$ true\nexit code: 0"


async def test_run_command_reports_a_non_zero_exit(
    workspace: Path, settings: Settings
) -> None:
    result = await make_registry(workspace, settings).invoke("run_command", {"command": "false"})
    assert "exit code: 1" in result


async def test_run_command_captures_stdout_and_stderr(workspace: Path, settings: Settings) -> None:
    script = "import sys; print('OUT-LINE'); sys.stderr.write('ERR-LINE\\n')"
    result = await make_registry(workspace, settings).invoke(
        "run_command", {"command": command(script)}
    )
    assert "--- stdout ---" in result
    assert "OUT-LINE" in result
    assert "--- stderr ---" in result
    assert "ERR-LINE" in result
    assert "exit code: 0" in result


async def test_run_command_uses_the_workspace_as_cwd(workspace: Path, settings: Settings) -> None:
    result = await make_registry(workspace, settings).invoke(
        "run_command", {"command": command("import os; print(os.getcwd())")}
    )
    assert str(workspace) in result


async def test_run_command_kills_on_timeout_and_says_so(
    workspace: Path, settings: Settings
) -> None:
    registry = make_registry(workspace, settings)
    started = time.monotonic()
    result = await registry.invoke("run_command", {"command": SLEEP_COMMAND, "timeout": 1})
    elapsed = time.monotonic() - started
    assert "TIMED OUT after 1s and was killed" in result
    assert "no exit code" in result
    assert elapsed < 5.0


async def test_run_command_timeout_defaults_to_the_settings_request_timeout(
    workspace: Path,
) -> None:
    settings = build_settings(workspace, request_timeout=1)
    registry = make_registry(workspace, settings)
    started = time.monotonic()
    result = await registry.invoke("run_command", {"command": SLEEP_COMMAND})
    assert "TIMED OUT after 1s and was killed" in result
    assert time.monotonic() - started < 5.0


async def test_run_command_refuses_an_empty_command(workspace: Path, settings: Settings) -> None:
    result = await make_registry(workspace, settings).invoke("run_command", {"command": "   "})
    assert result.startswith("ERROR:")
    assert "command must not be empty" in result


async def test_run_command_refuses_a_non_numeric_timeout(
    workspace: Path, settings: Settings
) -> None:
    result = await make_registry(workspace, settings).invoke(
        "run_command", {"command": "true", "timeout": "soon"}
    )
    assert result.startswith("ERROR:")
    assert "'timeout' must be a number" in result


async def test_run_command_event_payload_is_bounded(
    workspace: Path, settings: Settings, bus: EventBus
) -> None:
    script = "import sys; sys.stdout.write('y' * 100000)"
    registry = make_registry(workspace, settings, bus)
    await registry.invoke("run_command", {"command": command(script)})
    event = events_of(bus, EventType.TOOL_CALL)[0]
    assert event.data["exit_code"] == 0
    assert event.data["timed_out"] is False
    assert len(event.data["command"]) <= 100
    assert json.loads(json.dumps(event.to_dict()))


# ------------------------------------------------------ invoke never raises
async def test_invoke_reports_an_unknown_tool(workspace: Path, settings: Settings) -> None:
    result = await make_registry(workspace, settings).invoke("teleport", {})
    assert result.startswith("ERROR: unknown tool 'teleport'")
    assert "read_file" in result


async def test_invoke_reports_a_none_tool_name(workspace: Path, settings: Settings) -> None:
    result = await make_registry(workspace, settings).invoke(cast("str", None), {})
    assert result.startswith("ERROR: unknown tool None")


async def test_invoke_reports_missing_required_arguments(
    workspace: Path, settings: Settings
) -> None:
    registry = make_registry(workspace, settings)
    assert "missing required argument 'path'" in await registry.invoke("read_file", {})
    assert "missing required argument 'content'" in await registry.invoke(
        "write_file", {"path": "a.txt"}
    )
    assert "missing required argument 'old_string'" in await registry.invoke(
        "edit_file", {"path": "a.txt", "new_string": "x"}
    )
    assert "missing required argument 'query'" in await registry.invoke("search", {})


async def test_invoke_reports_wrong_argument_types(workspace: Path, settings: Settings) -> None:
    registry = make_registry(workspace, settings)
    nested = await registry.invoke("read_file", {"path": {"nested": "object"}})
    assert nested.startswith("ERROR:")
    assert "'path' must be a string, got dict" in nested
    listed = await registry.invoke("write_file", {"path": "a.txt", "content": ["nope"]})
    assert "'content' must be a string, got list" in listed


async def test_invoke_reports_an_empty_string_path(workspace: Path, settings: Settings) -> None:
    (workspace / "a.txt").write_text("", encoding="utf-8")
    result = await make_registry(workspace, settings).invoke("read_file", {"path": ""})
    assert result.startswith("ERROR:")
    assert "is a directory, not a file" in result


async def test_invoke_reports_a_directory_where_a_file_is_expected(
    workspace: Path, settings: Settings
) -> None:
    (workspace / "pkg").mkdir()
    result = await make_registry(workspace, settings).invoke("read_file", {"path": "pkg/"})
    assert result.startswith("ERROR:")
    assert "is a directory, not a file" in result


async def test_invoke_survives_non_dict_arguments(workspace: Path, settings: Settings) -> None:
    result = await make_registry(workspace, settings).invoke(
        "list_files", cast("dict[str, Any]", ["not", "a", "mapping"])
    )
    assert "No files under . match" in result


# ------------------------------------------------------- SANDBOX ESCAPE (security)
FILE_ESCAPES: dict[str, Callable[[Path], str]] = {
    "traversal": lambda outside: f"../{outside.name}/secret.txt",
    "absolute": lambda outside: str(outside / "secret.txt"),
    "symlink_to_file": lambda outside: "link.txt",
    "symlink_to_dir": lambda outside: "outdir/secret.txt",
    "normalised": lambda outside: f"nested/../../{outside.name}/secret.txt",
}
DIR_ESCAPES: dict[str, Callable[[Path], str]] = {
    "traversal": lambda outside: f"../{outside.name}",
    "absolute": lambda outside: str(outside),
    "symlink_to_dir": lambda outside: "outdir",
    "normalised": lambda outside: f"nested/../../{outside.name}",
}
FILE_TOOLS = ("read_file", "write_file", "edit_file")
DIR_TOOLS = ("list_files", "search")


def escape_arguments(tool: str, target: str) -> dict[str, Any]:
    """Arguments that would succeed if the path reached its target."""
    if tool == "read_file":
        return {"path": target}
    if tool == "write_file":
        return {"path": target, "content": "PWNED"}
    if tool == "edit_file":
        return {"path": target, "old_string": SECRET, "new_string": "PWNED"}
    if tool == "list_files":
        return {"path": target}
    return {"query": SECRET, "path": target}


@pytest.mark.parametrize("tool", FILE_TOOLS)
@pytest.mark.parametrize("vector", sorted(FILE_ESCAPES))
async def test_sandbox_refuses_to_read_a_file_outside_the_workspace(
    tool: str,
    vector: str,
    escape_workspace: Path,
    outside: Path,
    settings: Settings,
) -> None:
    target = FILE_ESCAPES[vector](outside)
    result = await make_registry(escape_workspace, settings).invoke(
        tool, escape_arguments(tool, target)
    )
    assert result.startswith("ERROR:")
    assert "outside the workspace sandbox" in result
    assert target in result
    assert SECRET not in result
    assert (outside / "secret.txt").read_text(encoding="utf-8") == SECRET_BODY
    assert sorted(entry.name for entry in outside.iterdir()) == ["secret.txt"]


@pytest.mark.parametrize("tool", DIR_TOOLS)
@pytest.mark.parametrize("vector", sorted(DIR_ESCAPES))
async def test_sandbox_refuses_to_walk_a_directory_outside_the_workspace(
    tool: str,
    vector: str,
    escape_workspace: Path,
    outside: Path,
    settings: Settings,
) -> None:
    target = DIR_ESCAPES[vector](outside)
    result = await make_registry(escape_workspace, settings).invoke(
        tool, escape_arguments(tool, target)
    )
    assert result.startswith("ERROR:")
    assert "outside the workspace sandbox" in result
    assert target in result
    assert SECRET not in result
    assert sorted(entry.name for entry in outside.iterdir()) == ["secret.txt"]


async def test_search_skips_symlinks_that_leave_the_sandbox(
    escape_workspace: Path, settings: Settings
) -> None:
    (escape_workspace / "src").mkdir()
    (escape_workspace / "src" / "keep.txt").write_text("alpha\n", encoding="utf-8")
    result = await make_registry(escape_workspace, settings).invoke(
        "search", {"query": SECRET, "path": "."}
    )
    assert result.startswith("No matches")
    assert "secret.txt" not in result
    assert "link.txt" not in result
    assert "outdir" not in result


async def test_list_files_skips_symlinks_that_leave_the_sandbox(
    escape_workspace: Path, settings: Settings
) -> None:
    (escape_workspace / "src").mkdir()
    (escape_workspace / "src" / "keep.txt").write_text("alpha\n", encoding="utf-8")
    result = await make_registry(escape_workspace, settings).invoke("list_files", {"path": "."})
    assert result == "src/keep.txt"


async def test_a_symlink_that_stays_inside_the_sandbox_is_allowed(
    workspace: Path, settings: Settings
) -> None:
    inner = workspace / "inner"
    inner.mkdir()
    (inner / "real.txt").write_text("IN-SANDBOX\n", encoding="utf-8")
    (workspace / "inside.txt").symlink_to(inner / "real.txt")
    (workspace / "inside_dir").symlink_to(inner, target_is_directory=True)
    registry = make_registry(workspace, settings)
    assert "IN-SANDBOX" in await registry.invoke("read_file", {"path": "inside.txt"})
    assert "inner/real.txt" in await registry.invoke("list_files", {"path": "inside_dir"})


async def test_write_through_a_relative_traversal_is_refused(
    workspace: Path, settings: Settings
) -> None:
    nested = workspace / "pkg" / "deep"
    nested.mkdir(parents=True)
    result = await make_registry(workspace, settings).invoke(
        "write_file", {"path": "pkg/deep/../../../escaped.py", "content": "nope"}
    )
    assert result.startswith("ERROR:")
    assert "escaped.py" in result
    assert not (workspace.parent / "escaped.py").exists()

# =========================================================== WORKER AGENT
def write_round(content: str = "print('hello')\n", path: str = "greeting.py") -> list[ToolCall]:
    return [ToolCall(id="t1", name="write_file", arguments={"path": path, "content": content})]


async def run_worker(
    task: TaskModel,
    client: ScriptedLLMClient,
    settings: Settings,
    bus: EventBus | None = None,
) -> str:
    worker = WorkerAgent(task=task, client=as_llm_client(client), settings=settings, bus=bus)
    return await worker.run()


async def test_worker_drives_the_tool_loop_and_writes_a_real_file(
    workspace: Path, make_client: Callable[..., ScriptedLLMClient], settings: Settings
) -> None:
    client = make_client(
        tool_rounds=[write_round(), []],
        text="Added greeting.py so the task prints a greeting.",
    )
    task = make_task()
    summary = await run_worker(task, client, settings)
    assert (workspace / "greeting.py").read_text(encoding="utf-8") == "print('hello')\n"
    assert task.status is TaskStatus.COMPLETED
    assert task.result == summary
    assert summary == (
        "Added greeting.py so the task prints a greeting.\n"
        "Files changed: greeting.py\n"
        "Tool results collected: 1"
    )
    assert task.attempts == 1


async def test_worker_feeds_tool_results_back_into_the_next_round(
    workspace: Path, make_client: Callable[..., ScriptedLLMClient], settings: Settings
) -> None:
    client = make_client(tool_rounds=[write_round(), []], text="done")
    await run_worker(make_task(), client, settings)
    rounds = [call for call in client.calls if call["method"] == "complete_with_tools"]
    assert len(rounds) == 2
    assert "[write_file ->]" not in rounds[0]["prompt"]
    assert "[write_file ->]" in rounds[1]["prompt"]
    assert "Wrote 15 bytes to greeting.py" in rounds[1]["prompt"]


async def test_worker_passes_the_tool_contract_to_the_client(
    workspace: Path, make_client: Callable[..., ScriptedLLMClient], settings: Settings
) -> None:
    client = make_client(tool_rounds=[[],])
    await run_worker(make_task(), client, settings)
    call = client.calls[0]
    assert call["method"] == "complete_with_tools"
    assert call["max_iterations"] == 1
    assert call["system"] == WORKER_SYSTEM_PROMPT
    assert call["model"] == settings.worker_model
    assert call["tools"] == [
        "read_file",
        "write_file",
        "edit_file",
        "list_files",
        "search",
        "run_command",
    ]
    assert len(call["tools"]) == 6


async def test_worker_orients_the_model_with_the_workspace_listing(
    workspace: Path, make_client: Callable[..., ScriptedLLMClient], settings: Settings
) -> None:
    (workspace / "existing.txt").write_text("alpha\n", encoding="utf-8")
    client = make_client(tool_rounds=[[],])
    await run_worker(make_task(), client, settings)
    prompt = client.calls[0]["prompt"]
    assert "Workspace contents (workspace-relative paths):" in prompt
    assert "existing.txt" in prompt
    assert "Task 1: write greeting.py" in prompt
    assert "Target files: greeting.py" in prompt
    assert str(workspace) in prompt


async def test_worker_reports_an_empty_tool_script_as_a_plain_summary(
    workspace: Path, make_client: Callable[..., ScriptedLLMClient], settings: Settings
) -> None:
    client = make_client(tool_rounds=[[],], text="Nothing to do.")
    summary = await run_worker(make_task(), client, settings)
    assert summary == "Nothing to do.\nFiles changed: none"


async def test_worker_says_incomplete_when_the_iteration_cap_is_hit(
    workspace: Path, make_client: Callable[..., ScriptedLLMClient]
) -> None:
    settings = build_settings(workspace, max_tool_iterations=2)
    client = make_client(tool_rounds=[write_round(), write_round()], text="working on it")
    task = make_task()
    summary = await run_worker(task, client, settings)
    assert summary.startswith("INCOMPLETE: the tool iteration limit was reached")
    assert "NOT" in summary and "finished" in summary
    assert "Files changed: greeting.py" in summary
    assert task.status is TaskStatus.COMPLETED


async def test_worker_reports_the_tools_it_actually_collected(
    workspace: Path, make_client: Callable[..., ScriptedLLMClient]
) -> None:
    settings = build_settings(workspace, max_tool_iterations=3)
    client = make_client(tool_rounds=[write_round(), write_round(), []], text="done")
    summary = await run_worker(make_task(), client, settings)
    assert "Tool results collected: 2" in summary
    assert "INCOMPLETE" not in summary


async def test_worker_surfaces_an_error_stop_reason(
    workspace: Path, settings: Settings
) -> None:
    client = _ErrorStopClient(settings)
    summary = await run_worker(make_task(), client, settings)
    assert "stop_reason='error'" in summary
    assert "RateLimitError" in summary
    assert "INCOMPLETE" not in summary


async def test_worker_fails_when_the_client_has_no_tool_support(workspace: Path) -> None:
    settings = build_settings(workspace)
    task = make_task()
    client = cast("LLMClient", _NoToolClient(settings))
    worker = WorkerAgent(task=task, client=client, settings=settings)
    summary = await worker.run()
    assert task.status is TaskStatus.FAILED
    assert "has no complete_with_tools" in summary
    assert "AttributeError" not in summary


async def test_worker_fails_when_the_workspace_is_unusable(workspace: Path) -> None:
    blocker = workspace / "blocker.txt"
    blocker.write_text("not a directory\n", encoding="utf-8")
    settings = build_settings(blocker)
    client = ScriptedLLMClient(settings, tool_rounds=[[],])
    task = make_task()
    summary = await run_worker(task, client, settings)
    assert task.status is TaskStatus.FAILED
    assert "is not an existing directory" in summary
    assert client.calls == []


async def test_worker_is_not_confused_by_a_cancelled_loop(workspace: Path) -> None:
    settings = build_settings(workspace)

    class _Cancelling(ScriptedLLMClient):
        async def complete_with_tools(self, prompt: str, tools: Any, **kwargs: Any) -> ToolResult:
            raise asyncio.CancelledError

    task = make_task()
    worker = WorkerAgent(
        task=task, client=as_llm_client(_Cancelling(settings)), settings=settings
    )
    with pytest.raises(asyncio.CancelledError):
        await worker.run()
    assert task.status is TaskStatus.FAILED
    assert task.result == "Task cancelled before it finished."


async def test_worker_emits_task_start_and_task_end(
    workspace: Path, make_client: Callable[..., ScriptedLLMClient], settings: Settings
) -> None:
    client = make_client(tool_rounds=[write_round(), []], text="done")
    bus = EventBus(history=settings.max_event_history)
    task = make_task()
    await run_worker(task, client, settings, bus)
    started = events_of(bus, EventType.TASK_START)
    ended = events_of(bus, EventType.TASK_END)
    assert len(started) == 1 and len(ended) == 1
    assert started[0].data == {
        "task_id": "1",
        "files": ["greeting.py"],
        "attempt": 1,
    }
    assert ended[0].data == {
        "task_id": "1",
        "status": "completed",
        "changed": ["greeting.py"],
    }


async def test_worker_emits_one_tool_call_event_per_tool(
    workspace: Path, make_client: Callable[..., ScriptedLLMClient], settings: Settings
) -> None:
    client = make_client(tool_rounds=[write_round(), []], text="done")
    bus = EventBus(history=settings.max_event_history)
    await run_worker(make_task(), client, settings, bus)
    tool_events = events_of(bus, EventType.TOOL_CALL)
    assert [event.data["tool"] for event in tool_events] == ["list_files", "write_file"]
    assert tool_events[1].data == {
        "path": "greeting.py",
        "bytes": 15,
        "tool": "write_file",
        "ok": True,
        "ms": tool_events[1].data["ms"],
    }


async def test_worker_events_are_json_serialisable(
    workspace: Path, make_client: Callable[..., ScriptedLLMClient], settings: Settings
) -> None:
    (workspace / "big.txt").write_text("payload-body\n" * 5_000, encoding="utf-8")
    client = make_client(
        tool_rounds=[
            [
                ToolCall(id="a", name="read_file", arguments={"path": "big.txt"}),
                ToolCall(id="b", name="search", arguments={"query": "payload-body"}),
            ],
            [],
        ],
        text="read the file",
    )
    bus = EventBus(history=settings.max_event_history)
    await run_worker(make_task(), client, settings, bus)
    dumped = json.dumps([event.to_dict() for event in bus.history()])
    assert len(dumped) < 20_000


async def test_worker_event_payloads_never_carry_file_content(
    workspace: Path, make_client: Callable[..., ScriptedLLMClient], settings: Settings
) -> None:
    marker = "ZZ-CONTENT-MARKER-ZZ"
    (workspace / "big.txt").write_text(marker + "\n", encoding="utf-8")
    client = make_client(
        tool_rounds=[
            [ToolCall(id="a", name="read_file", arguments={"path": "big.txt"})],
            [],
        ],
        text="read it",
    )
    bus = EventBus(history=settings.max_event_history)
    await run_worker(make_task(), client, settings, bus)
    for event in bus.history():
        assert marker not in event.message
        assert marker not in json.dumps(event.data)
    assert len(json.dumps(events_of(bus, EventType.TOOL_CALL)[-1].to_dict())) < 1_000


async def test_two_workers_cannot_see_each_others_files(
    tmp_path: Path, make_client: Callable[..., ScriptedLLMClient]
) -> None:
    root = tmp_path.resolve()
    left, right = root / "left", root / "right"
    left.mkdir()
    right.mkdir()

    async def worker_for(directory: Path, content: str) -> tuple[TaskModel, str]:
        settings = build_settings(directory)
        client = ScriptedLLMClient(settings, tool_rounds=[write_round(content)], text="wrote it")
        task = make_task(files=[f"{directory.name}/own.py"])
        return task, await run_worker(task, client, settings)

    left_task, before = await worker_for(left, "LEFT-SIDE\n")
    right_task, after = await worker_for(right, "RIGHT-SIDE\n")
    assert (left / "greeting.py").read_text(encoding="utf-8") == "LEFT-SIDE\n"
    assert (right / "greeting.py").read_text(encoding="utf-8") == "RIGHT-SIDE\n"
    assert left_task.status is TaskStatus.COMPLETED
    assert right_task.status is TaskStatus.COMPLETED
    assert "LEFT-SIDE" not in after
    assert "RIGHT-SIDE" not in before
    assert "Files changed: greeting.py" in before
    assert "Files changed: greeting.py" in after


async def test_worker_runs_without_a_bus(
    workspace: Path, make_client: Callable[..., ScriptedLLMClient], settings: Settings
) -> None:
    client = make_client(tool_rounds=[write_round(), []], text="quiet")
    task = make_task()
    summary = await run_worker(task, client, settings)
    assert task.status is TaskStatus.COMPLETED
    assert "quiet" in summary


# ========================================================== PLANNER AGENT
async def plan_of(
    settings: Settings, payload: Any, bus: EventBus | None = None
) -> Plan:
    client = ScriptedLLMClient(settings, plan=payload)
    return await PlannerAgent(client=as_llm_client(client), settings=settings, bus=bus).plan(GOAL)


async def plan_ids(settings: Settings, payload: Any) -> list[str]:
    return [task.id for task in (await plan_of(settings, payload)).tasks]


async def test_planner_passes_the_system_prompt_separately(
    settings: Settings, make_client: Callable[..., ScriptedLLMClient]
) -> None:
    client = make_client(plan={"tasks": [], "rationale": "nothing to do"})
    await PlannerAgent(client=as_llm_client(client), settings=settings).plan(GOAL)
    call = client.calls[0]
    assert call["method"] == "complete_json"
    assert call["system"] == PLANNER_PROMPT
    assert PLANNER_PROMPT not in call["prompt"]
    assert call["prompt"].startswith(f"Goal:\n{GOAL}\n")
    assert call["model"] == settings.planner_model


async def test_planner_synthesises_a_missing_id_from_the_index(settings: Settings) -> None:
    payload = {"tasks": [{"description": "first"}, {"description": "second"}]}
    assert await plan_ids(settings, payload) == ["1", "2"]


async def test_planner_synthesises_an_id_for_a_bare_string_task(settings: Settings) -> None:
    plan = await plan_of(settings, {"tasks": ["just a sentence"]})
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "just a sentence"


async def test_planner_coerces_a_numeric_id_to_a_string(settings: Settings) -> None:
    payload = {
        "tasks": [
            {"id": 7, "description": "seventh"},
            {"id": 8, "description": "eighth", "dependencies": [7, "7"]},
        ]
    }
    plan = await plan_of(settings, payload)
    assert [task.id for task in plan.tasks] == ["7", "8"]
    # The numeric entry is dropped: `_string_list` keeps only real strings. The coerced
    # id still matches the string form the model is told to send.
    assert plan.tasks[1].dependencies == ["7"]


async def test_planner_replaces_a_blank_id(settings: Settings) -> None:
    assert await plan_ids(settings, {"tasks": [{"id": "   ", "description": "blank"}]}) == ["1"]


async def test_planner_synthesises_a_missing_description(settings: Settings) -> None:
    plan = await plan_of(settings, {"tasks": [{"id": "a"}]})
    assert plan.tasks[0].description == "Task a: the planner returned no description."


async def test_planner_stringifies_a_non_string_description(settings: Settings) -> None:
    plan = await plan_of(settings, {"tasks": [{"id": "a", "description": 42}]})
    assert plan.tasks[0].description == "42"


async def test_planner_truncates_an_over_long_description(settings: Settings) -> None:
    plan = await plan_of(settings, {"tasks": [{"id": "a", "description": "x" * 900}]})
    assert len(plan.tasks[0].description) < 500
    assert "chars truncated" in plan.tasks[0].description


async def test_planner_coerces_a_non_list_files_field(settings: Settings) -> None:
    plan = await plan_of(
        settings,
        {
            "tasks": [
                {"id": "a", "files": "a.py"},
                {"id": "b", "files": {"nested": 1}},
                {"id": "c", "files": None},
                {"id": "d", "files": ["ok.py", 7, None, "  spaced.py  "]},
            ]
        },
    )
    assert [task.files for task in plan.tasks] == [[], [], [], ["ok.py", "spaced.py"]]


async def test_planner_drops_junk_dependency_entries(settings: Settings) -> None:
    plan = await plan_of(
        settings,
        {
            "tasks": [
                {"id": "a", "dependencies": ["  ", 3, None, {"x": 1}, "b", "b", True]},
                {"id": "b"},
                {"id": "c", "dependencies": "b"},
            ]
        },
    )
    assert plan.tasks[0].dependencies == ["b"]
    assert plan.tasks[2].dependencies == []


async def test_planner_makes_duplicate_ids_unique(settings: Settings) -> None:
    payload = {
        "tasks": [
            {"id": "dup", "description": "first"},
            {"id": "dup", "description": "second"},
            {"id": "dup", "description": "third"},
            {"id": "dup-2", "description": "fourth"},
        ]
    }
    assert await plan_ids(settings, payload) == ["dup", "dup-2", "dup-3", "dup-2-2"]


async def test_planner_rewrites_dependencies_across_a_rename(settings: Settings) -> None:
    payload = {
        "tasks": [
            {"id": "dup", "description": "first"},
            {"id": "dup", "description": "second", "dependencies": ["dup"]},
        ]
    }
    plan = await plan_of(settings, payload)
    assert [task.id for task in plan.tasks] == ["dup", "dup-2"]
    assert plan.tasks[0].dependencies == []
    assert plan.tasks[1].dependencies == ["dup"]


async def test_planner_strips_a_self_dependency(settings: Settings) -> None:
    plan = await plan_of(
        settings, {"tasks": [{"id": "solo", "dependencies": ["solo", "solo"]}]}
    )
    assert plan.tasks[0].dependencies == []


async def test_planner_drops_unknown_keys(settings: Settings) -> None:
    plan = await plan_of(
        settings,
        {
            "tasks": [
                {
                    "id": "a",
                    "description": "d",
                    "status": "completed",
                    "attempts": 99,
                    "result": "smuggled",
                    "hallucinated": {"deep": 1},
                }
            ]
        },
    )
    task = plan.tasks[0]
    assert task.status.value == "pending"
    assert task.attempts == 0
    assert task.result is None
    assert not hasattr(task, "hallucinated")


async def test_planner_stringifies_a_non_string_rationale(settings: Settings) -> None:
    plan = await plan_of(settings, {"tasks": [], "rationale": 17})
    assert plan.rationale == "17"


async def test_planner_accepts_an_empty_task_list(settings: Settings) -> None:
    plan = await plan_of(settings, {"tasks": [], "rationale": "goal already met"})
    assert plan.tasks == []
    assert plan.rationale == "goal already met"


async def test_planner_does_not_truncate_to_max_parallel_tasks(settings: Settings) -> None:
    assert settings.max_parallel_tasks < 10
    payload = {"tasks": [{"id": str(i), "description": f"task {i}"} for i in range(1, 11)]}
    assert len(await plan_ids(settings, payload)) == 10


@pytest.mark.parametrize(
    "payload",
    [["a", "list"], "a bare string", 42],
    ids=["list", "str", "int"],
)
async def test_planner_rejects_a_non_object_payload(settings: Settings, payload: Any) -> None:
    with pytest.raises(LLMResponseError, match="must be a JSON object"):
        await plan_of(settings, payload)


class _NoneJsonClient(ScriptedLLMClient):
    """A client whose `complete_json` returns a JSON `null`."""

    async def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_attempts: int = 2,
    ) -> Any:
        self._record("complete_json", prompt, system, model, max_attempts=max_attempts)
        return None


async def test_planner_rejects_a_null_payload(workspace: Path) -> None:
    settings = build_settings(workspace)
    planner = PlannerAgent(client=as_llm_client(_NoneJsonClient(settings)), settings=settings)
    with pytest.raises(LLMResponseError, match="must be a JSON object, got NoneType"):
        await planner.plan(GOAL)


async def test_planner_rejects_a_missing_tasks_key(settings: Settings) -> None:
    with pytest.raises(LLMResponseError, match="missing a 'tasks' array"):
        await plan_of(settings, {"rationale": "forgot the tasks"})


@pytest.mark.parametrize(
    "payload",
    [{"tasks": "not a list"}, {"tasks": {"id": "a"}}, {"tasks": None}],
    ids=["str", "dict", "none"],
)
async def test_planner_rejects_a_non_list_tasks_field(settings: Settings, payload: Any) -> None:
    with pytest.raises(LLMResponseError, match="missing a 'tasks' array"):
        await plan_of(settings, payload)


async def test_planner_rejects_a_json_string_body(settings: Settings) -> None:
    with pytest.raises(LLMResponseError):
        await plan_of(settings, '{"tasks": ["not an object"]}')


async def test_planner_wraps_a_raw_json_decode_error(workspace: Path) -> None:
    settings = build_settings(workspace)
    client = _LeakyDecoderClient(settings)
    planner = PlannerAgent(client=as_llm_client(client), settings=settings)
    with pytest.raises(LLMResponseError) as caught:
        await planner.plan(GOAL)
    assert not isinstance(caught.value, json.JSONDecodeError)
    assert "was not valid JSON" in str(caught.value)


async def test_planner_propagates_a_transport_error_unchanged(workspace: Path) -> None:
    settings = build_settings(workspace)
    client = ScriptedLLMClient(settings, error=RuntimeError("provider exploded"))
    planner = PlannerAgent(client=as_llm_client(client), settings=settings)
    with pytest.raises(RuntimeError, match="provider exploded"):
        await planner.plan(GOAL)


async def test_planner_emits_a_json_serialisable_plan_event(
    settings: Settings, bus: EventBus
) -> None:
    payload = {
        "tasks": [{"id": "1", "description": "do it", "files": ["a.py"], "dependencies": ["2"]},
                  {"id": "2", "description": "do it first"}],
        "rationale": "two steps",
    }
    plan = await plan_of(settings, payload, bus)
    events = events_of(bus, EventType.PLAN)
    assert len(events) == 1
    assert events[0].data["task_count"] == 2
    assert events[0].data["tasks"][0] == {
        "id": "1",
        "description": "do it",
        "files": 1,
        "dependencies": 1,
    }
    assert json.loads(json.dumps(events[0].to_dict()))["data"]["rationale"] == "two steps"
    assert events_of(bus, EventType.ERROR) == []
    assert plan.tasks[1].id == "2"


async def test_planner_emits_an_error_event_on_failure(
    settings: Settings, bus: EventBus
) -> None:
    with pytest.raises(LLMResponseError):
        await plan_of(settings, {"rationale": "no tasks here"}, bus)
    errors = events_of(bus, EventType.ERROR)
    assert len(errors) == 1
    assert "Planner failed" in errors[0].message
    assert "missing a 'tasks' array" in errors[0].data["error"]
    assert json.loads(json.dumps(errors[0].to_dict()))["type"] == "error"


async def test_planner_works_without_a_bus(settings: Settings) -> None:
    plan = await plan_of(settings, {"tasks": [{"id": "a", "description": "d"}]})
    assert [task.id for task in plan.tasks] == ["a"]


# =========================================================== TESTER AGENT
FALSE_PASS = {"passed": True, "summary": "all good", "failures": []}


async def run_tester(
    settings: Settings,
    payload: Any = DEFAULT_TEST,
    tasks: list[TaskModel] | None = None,
    bus: EventBus | None = None,
) -> tuple[TestResult, ScriptedLLMClient]:
    client = ScriptedLLMClient(settings, test=payload)
    agent = tester_agent.TesterAgent(client=as_llm_client(client), settings=settings, bus=bus)
    result = await agent.run(GOAL, tasks or [make_task()])
    return result, client


def failing_pytest_workspace(root: Path) -> None:
    (root / "test_broken.py").write_text(
        "def test_definitely_fails():\n    assert 1 == 2\n", encoding="utf-8"
    )


# --- regression #12: the model cannot talk its way past a real non-zero exit -------
async def test_tester_overrides_a_false_pass_on_a_real_failing_suite(workspace: Path) -> None:
    failing_pytest_workspace(workspace)
    settings = build_settings(workspace, test_command="pytest -q -p no:cacheprovider")
    result, _ = await run_tester(settings, FALSE_PASS)
    assert result.passed is False
    assert result.commands[0] == "pytest -q -p no:cacheprovider"
    assert any(
        "pytest -q -p no:cacheprovider" in failure and "exited with code 1" in failure
        for failure in result.failures
    )
    assert "command exit statuses contradict the model verdict" in result.summary


async def test_tester_false_pass_does_not_mutate_the_scripted_payload(workspace: Path) -> None:
    failing_pytest_workspace(workspace)
    settings = build_settings(workspace, test_command=FAILING_COMMAND)
    payload = dict(FALSE_PASS)
    result, _ = await run_tester(settings, payload)
    assert result.passed is False
    assert payload == {"passed": True, "summary": "all good", "failures": []}


async def test_tester_keeps_a_false_pass_alone_when_every_command_exited_zero(
    workspace: Path,
) -> None:
    settings = build_settings(workspace, test_command=PASSING_COMMAND)
    result, _ = await run_tester(settings, DEFAULT_TEST)
    assert result.passed is True
    assert result.failures == []
    assert "contradict" not in result.summary


async def test_tester_does_not_manufacture_a_failure_the_ground_truth_does_not_show(
    workspace: Path,
) -> None:
    settings = build_settings(workspace, test_command=PASSING_COMMAND)
    payload = {"passed": False, "summary": "no tests ran", "failures": ["nothing collected"]}
    result, _ = await run_tester(settings, payload)
    assert result.passed is False
    assert result.failures == ["nothing collected"]
    assert "contradict" not in result.summary


async def test_tester_treats_an_unusable_reply_as_a_failure(workspace: Path) -> None:
    settings = build_settings(workspace, test_command=PASSING_COMMAND)
    result, _ = await run_tester(settings, "not json at all")
    assert result.passed is False
    assert "not usable JSON" in result.summary
    assert result.failures


@pytest.mark.parametrize("payload", [{"passed": "true"}, {"passed": 1}, {}, [1, 2]])
async def test_tester_rejects_a_non_boolean_passed(workspace: Path, payload: Any) -> None:
    settings = build_settings(workspace, test_command=PASSING_COMMAND)
    result, _ = await run_tester(settings, payload)
    assert result.passed is False
    assert "not usable JSON" in result.summary


async def test_tester_treats_an_unparseable_reply_as_a_failure(workspace: Path) -> None:
    settings = build_settings(workspace, test_command=PASSING_COMMAND)
    client = ScriptedLLMClient(settings, error=LLMResponseError("Model output was not valid JSON"))
    agent = tester_agent.TesterAgent(client=as_llm_client(client), settings=settings)
    result = await agent.run(GOAL, [make_task()])
    assert result.passed is False
    assert "not parseable JSON" in result.summary
    assert result.commands[0] == PASSING_COMMAND


async def test_tester_propagates_an_unexpected_provider_fault(workspace: Path) -> None:
    settings = build_settings(workspace, test_command=PASSING_COMMAND)
    client = ScriptedLLMClient(settings, error=LLMConfigurationError("no API key"))
    agent = tester_agent.TesterAgent(client=as_llm_client(client), settings=settings)
    with pytest.raises(LLMConfigurationError, match="no API key"):
        await agent.run(GOAL, [make_task()])


# --- command resolution ----------------------------------------------------------
async def test_tester_honours_all_three_command_overrides(workspace: Path) -> None:
    settings = build_settings(
        workspace,
        test_command=PASSING_COMMAND,
        lint_command=command("import sys; sys.exit(3)"),
        typecheck_command=command("import sys; sys.exit(4)"),
    )
    result, _ = await run_tester(settings, FALSE_PASS)
    assert result.commands == [PASSING_COMMAND, settings.lint_command, settings.typecheck_command]
    assert result.passed is False
    assert any("exited with code 3" in failure for failure in result.failures)
    assert any("exited with code 4" in failure for failure in result.failures)


async def test_tester_deduplicates_an_override_that_repeats_a_command(
    workspace: Path,
) -> None:
    settings = build_settings(
        workspace, test_command=PASSING_COMMAND, lint_command=PASSING_COMMAND
    )
    result, _ = await run_tester(settings, DEFAULT_TEST)
    assert result.commands == [PASSING_COMMAND]
    assert result.passed is True


async def test_tester_runs_the_test_and_lint_commands_it_detects(workspace: Path) -> None:
    (workspace / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n", encoding="utf-8")
    (workspace / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (workspace / "bad_lint.py").write_text("import os\n", encoding="utf-8")
    (workspace / "bad_types.py").write_text(
        "def boom() -> int:\n    return 'not an int'\n", encoding="utf-8"
    )
    settings = build_settings(workspace)
    result, client = await run_tester(settings, DEFAULT_TEST)
    assert len(result.commands) == 3
    assert result.commands[0] == "pytest -q"
    assert result.commands[1] == "ruff check ."
    assert result.commands[2] == "mypy ."
    assert result.passed is False
    assert any("pytest -q exited with code 0" not in failure for failure in result.failures)
    assert any("ruff check . exited with code 1" in failure for failure in result.failures)
    assert any("mypy . exited with code 1" in failure for failure in result.failures)
    statuses = client.calls[0]["prompt"]
    assert "pytest -q -> exit code 0" in statuses
    assert "ruff check . -> exit code 1" in statuses
    assert "mypy . -> exit code 1" in statuses


async def test_tester_skips_verification_when_no_test_command_resolves(
    workspace: Path,
) -> None:
    settings = build_settings(workspace, test_command="   ")
    result, client = await run_tester(settings, FALSE_PASS)
    assert result.commands == []
    assert "verification skipped" in result.summary
    assert result.passed is True
    assert client.calls == []


async def test_tester_reports_a_missing_override_binary_without_raising(
    workspace: Path,
) -> None:
    settings = build_settings(workspace, test_command="definitely-not-a-real-binary")
    result, _ = await run_tester(settings, FALSE_PASS)
    assert result.commands[0] == "definitely-not-a-real-binary"
    assert result.passed is False
    assert any("could not be executed" in failure for failure in result.failures)
    assert any("FileNotFoundError" in failure for failure in result.failures)


async def test_tester_kills_a_command_that_overruns_the_timeout(workspace: Path) -> None:
    sleeper = shlex.join([sys.executable, "-c", "import time; time.sleep(30)"])
    settings = build_settings(workspace, test_command=sleeper, test_timeout=1)
    started = time.monotonic()
    result, _ = await run_tester(settings, FALSE_PASS)
    elapsed = time.monotonic() - started
    assert result.passed is False
    assert result.failures == [f"{sleeper} timed out and was killed"]
    assert "timed out" in result.summary or result.failures
    assert elapsed < 5.0


async def test_tester_runs_commands_in_the_resolved_workspace(workspace: Path) -> None:
    probe = command("import os; print(os.getcwd())")
    settings = build_settings(workspace, test_command=probe)
    result, client = await run_tester(settings, DEFAULT_TEST)
    assert result.passed is True
    assert str(workspace) in client.calls[0]["prompt"]


async def test_tester_caps_the_output_handed_to_the_model(workspace: Path) -> None:
    settings = build_settings(workspace, test_command=NOISE_COMMAND)
    result, client = await run_tester(settings, DEFAULT_TEST)
    assert result.passed is True
    prompt = client.calls[0]["prompt"]
    assert len(prompt) < 20_000
    assert "earlier characters omitted" in prompt
    assert "TAIL-MARKER" in prompt
    assert "x" * 15_000 not in prompt
    assert len(prompt) < NOISE_CHARACTERS // 10


async def test_tester_prompt_names_the_goal_the_tasks_and_every_status(
    workspace: Path,
) -> None:
    settings = build_settings(workspace, test_command=PASSING_COMMAND)
    _, client = await run_tester(settings, DEFAULT_TEST, tasks=[make_task()])
    prompt = client.calls[0]["prompt"]
    assert client.calls[0]["system"] == TESTER_PROMPT
    assert f"Goal:\n{GOAL}" in prompt
    assert "- 1: write greeting.py (files: greeting.py)" in prompt
    assert f"- {PASSING_COMMAND} -> exit code 0" in prompt


async def test_tester_emits_a_json_serialisable_test_result_event(
    workspace: Path, bus: EventBus
) -> None:
    settings = build_settings(workspace, test_command=FAILING_COMMAND)
    result, _ = await run_tester(settings, DEFAULT_TEST, bus=bus)
    events = events_of(bus, EventType.TEST_RESULT)
    assert len(events) == 1
    assert events[0].data["passed"] is False
    assert events[0].data["commands"] == result.commands
    assert events[0].data["failure_count"] == len(result.failures)
    assert json.loads(json.dumps([event.to_dict() for event in bus.history()]))
    errors = events_of(bus, EventType.ERROR)
    assert len(errors) == 1
    assert errors[0].data["status"] == "exit code 1"
