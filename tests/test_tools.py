"""Behavioural tests for :mod:`furrow.tools`.

Everything here runs against a real (temporary) directory but never touches the
network. The security-critical surface -- workspace path containment -- is
covered from three directions: a ``..`` escape, an absolute path outside the
root, and a symlink that points out of the workspace.

The module documents two failure styles for a refused path: ``_resolve`` raises
:class:`~furrow.errors.WorkspaceContainmentError`, and every public tool catches
it and returns ``ToolResult(ok=False)``. Both are asserted here, because the
tool-loop contract (``invoke`` "never raises") depends on the second.
"""

from __future__ import annotations

import os
import shlex
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from furrow.config import Settings
from furrow.errors import ToolError, WorkspaceContainmentError
from furrow.tools import ToolResult, WorkspaceTools

ALL_TOOL_NAMES = {"read_file", "write_file", "edit_file", "list_files", "run_command"}


def _make_tools(workspace: Path, **kwargs: Any) -> WorkspaceTools:
    """Build a WorkspaceTools rooted at ``workspace`` with test-friendly flags."""
    return WorkspaceTools(workspace, **kwargs)


def _dir_entries(path: Path) -> set[str]:
    return {entry.name for entry in path.iterdir()}


# --- construction ---


def test_root_is_resolved_workspace(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    assert tools.root == tmp_workspace.resolve()
    assert tools.root.is_absolute()


def test_root_resolves_a_relative_workspace(tmp_workspace: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace.parent)
    tools = WorkspaceTools(tmp_workspace.name)
    assert tools.root == tmp_workspace.resolve()


def test_names_lists_every_registered_tool(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    assert set(tools.names()) == ALL_TOOL_NAMES
    assert len(tools.names()) == len(ALL_TOOL_NAMES), "no duplicate registrations"


async def test_from_settings_carries_allow_shell_and_shell_timeout(
    make_settings: Callable[..., Settings], tmp_workspace: Path
) -> None:
    cfg = make_settings(allow_shell=True, shell_timeout=7)
    tools = WorkspaceTools.from_settings(cfg)
    assert tools.root == tmp_workspace.resolve()
    # allow_shell came from Settings, not the constructor default of False.
    allowed = await tools.invoke("run_command", {"command": "true"})
    assert allowed.ok is True

    locked = WorkspaceTools.from_settings(make_settings(allow_shell=False))
    refused = await locked.invoke("run_command", {"command": "true"})
    assert refused.ok is False
    assert "disabled" in (refused.error or "").lower()


def test_from_settings_with_none_uses_singleton() -> None:
    # cfg=None falls back to the module-level ``settings``; it must not raise and
    # must produce a workspace rooted at that singleton's workspace.
    from furrow.config import settings as global_settings

    tools = WorkspaceTools.from_settings(None)
    assert tools.root == Path(global_settings.workspace).resolve()


# --- write_file / read_file round trip ---


async def test_write_then_read_round_trips(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    written = await tools.invoke("write_file", {"path": "notes.txt", "content": "hello"})
    assert isinstance(written, ToolResult)
    assert written.ok is True
    assert written.error is None
    assert "notes.txt" in written.output

    read = await tools.invoke("read_file", {"path": "notes.txt"})
    assert read.ok is True
    assert read.output == "hello"
    assert (tmp_workspace / "notes.txt").read_text(encoding="utf-8") == "hello"


async def test_write_file_overwrites_and_leaves_no_temp_files(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    target = tmp_workspace / "code.py"

    first = await tools.invoke("write_file", {"path": "code.py", "content": "v1\n"})
    assert first.ok is True
    assert target.read_text(encoding="utf-8") == "v1\n"
    # The write is atomic: temp sibling + os.replace, so the only entry left is
    # the target itself.
    assert _dir_entries(tmp_workspace) == {"code.py"}

    second = await tools.invoke("write_file", {"path": "code.py", "content": "v2 longer\n"})
    assert second.ok is True
    assert target.read_text(encoding="utf-8") == "v2 longer\n"
    assert _dir_entries(tmp_workspace) == {"code.py"}
    assert not [name for name in _dir_entries(tmp_workspace) if ".tmp." in name]


async def test_write_file_creates_parent_directories(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    result = await tools.invoke(
        "write_file", {"path": "deep/nested/dir/file.txt", "content": "x"}
    )
    assert result.ok is True
    assert (tmp_workspace / "deep/nested/dir/file.txt").read_text(encoding="utf-8") == "x"


async def test_write_file_reports_utf8_byte_count(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    result = await tools.invoke("write_file", {"path": "u.txt", "content": "\u00e9"})
    assert result.ok is True
    # "e-acute" is 2 bytes in UTF-8 but 1 character.
    assert "2 bytes" in result.output


async def test_read_file_missing_file_returns_tool_result(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    result = await tools.invoke("read_file", {"path": "nope.txt"})
    assert result.ok is False
    assert result.error is not None
    assert "not found" in result.error.lower()


async def test_read_file_on_directory_returns_tool_result(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    (tmp_workspace / "adir").mkdir()
    result = await tools.invoke("read_file", {"path": "adir"})
    assert result.ok is False
    assert "director" in (result.error or "").lower()


async def test_output_is_truncated_to_max_output_chars(tmp_workspace: Path) -> None:
    tools = WorkspaceTools(tmp_workspace, max_output_chars=10)
    await tools.invoke("write_file", {"path": "big.txt", "content": "y" * 50})
    read = await tools.invoke("read_file", {"path": "big.txt"})
    assert read.ok is True
    assert read.output.count("y") == 10
    assert "truncated" in read.output


# --- edit_file ---


async def test_edit_file_unique_target_succeeds_and_reports_count(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    target = tmp_workspace / "app.py"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    result = await tools.invoke(
        "edit_file", {"path": "app.py", "old": "beta", "new": "BETA"}
    )
    assert result.ok is True
    assert result.error is None
    assert "1 replacement" in result.output
    assert target.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"
    assert _dir_entries(tmp_workspace) == {"app.py"}


async def test_edit_file_non_unique_target_is_refused_and_file_unchanged(
    tmp_workspace: Path,
) -> None:
    tools = _make_tools(tmp_workspace)
    target = tmp_workspace / "dup.py"
    original = "x = 1\nx = 1\n"
    target.write_text(original, encoding="utf-8")

    result = await tools.invoke("edit_file", {"path": "dup.py", "old": "x = 1", "new": "x = 2"})

    assert result.ok is False
    assert result.error is not None
    assert "unique" in result.error.lower()
    assert "2 occurrences" in result.error
    # The refusal must not have touched the file at all.
    assert target.read_text(encoding="utf-8") == original
    assert _dir_entries(tmp_workspace) == {"dup.py"}


async def test_edit_file_replace_all_replaces_every_occurrence(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    target = tmp_workspace / "many.txt"
    target.write_text("a b a b a", encoding="utf-8")

    result = await tools.invoke(
        "edit_file", {"path": "many.txt", "old": "a", "new": "z", "replace_all": True}
    )
    assert result.ok is True
    assert "3 replacement" in result.output
    assert target.read_text(encoding="utf-8") == "z b z b z"


async def test_edit_file_replace_all_still_needs_the_target(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    target = tmp_workspace / "many.txt"
    original = "a b a"
    target.write_text(original, encoding="utf-8")

    result = await tools.invoke(
        "edit_file",
        {"path": "many.txt", "old": "zzz", "new": "q", "replace_all": True},
    )
    assert result.ok is False
    assert result.error == "old_string not found"
    assert target.read_text(encoding="utf-8") == original


async def test_edit_file_missing_target_is_refused(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    target = tmp_workspace / "app.py"
    original = "alpha\n"
    target.write_text(original, encoding="utf-8")

    result = await tools.invoke(
        "edit_file", {"path": "app.py", "old": "missing-string", "new": "x"}
    )
    assert result.ok is False
    assert result.error is not None
    assert "not found" in result.error.lower()
    assert target.read_text(encoding="utf-8") == original


async def test_edit_file_on_missing_file_returns_tool_result(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    result = await tools.invoke(
        "edit_file", {"path": "ghost.py", "old": "a", "new": "b"}
    )
    assert result.ok is False
    assert "not found" in (result.error or "").lower()


async def test_edit_file_no_op_edit_is_refused(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    (tmp_workspace / "same.txt").write_text("unchanged", encoding="utf-8")
    result = await tools.invoke(
        "edit_file", {"path": "same.txt", "old": "unchanged", "new": "unchanged"}
    )
    assert result.ok is False
    assert result.error == "no changes made"


# --- list_files ---


async def test_list_files_is_recursive_and_skips_noise_directories(
    tmp_workspace: Path,
) -> None:
    tools = _make_tools(tmp_workspace)
    for relative in (
        "top.txt",
        "src/main.py",
        "src/pkg/__init__.py",
        "docs/guide.md",
        ".git/config",
        ".git/objects/deadbeef",
        "__pycache__/mod.cpython-310.pyc",
        "src/__pycache__/other.pyc",
        "node_modules/pkg/index.js",
        "furrow.egg-info/PKG-INFO",
        "keep_me.txt",
    ):
        path = tmp_workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    result = await tools.invoke("list_files", {})
    assert result.ok is True
    lines = [line for line in result.output.splitlines() if line]

    assert "top.txt" in lines
    assert "keep_me.txt" in lines
    assert os.path.join("src", "main.py") in lines
    assert os.path.join("src", "pkg", "__init__.py") in lines
    assert os.path.join("docs", "guide.md") in lines
    # Directories are reported with a trailing slash.
    assert "src/" in lines

    assert not [line for line in lines if ".git" in line]
    assert not [line for line in lines if "__pycache__" in line]
    assert not [line for line in lines if "node_modules" in line]
    assert not [line for line in lines if "egg-info" in line]
    assert not [line for line in lines if ".tmp." in line]


async def test_list_files_accepts_a_subdirectory(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    (tmp_workspace / "pkg").mkdir()
    (tmp_workspace / "pkg" / "mod.py").write_text("x", encoding="utf-8")
    (tmp_workspace / "outside.py").write_text("x", encoding="utf-8")

    result = await tools.invoke("list_files", {"path": "pkg"})
    assert result.ok is True
    assert result.output.splitlines() == ["mod.py"]


async def test_list_files_on_a_file_is_refused(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    (tmp_workspace / "file.txt").write_text("x", encoding="utf-8")
    result = await tools.invoke("list_files", {"path": "file.txt"})
    assert result.ok is False
    assert "not a directory" in (result.error or "").lower()


async def test_list_files_missing_directory_is_refused(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    result = await tools.invoke("list_files", {"path": "nowhere"})
    assert result.ok is False
    assert "not found" in (result.error or "").lower()


# --- workspace containment (security critical) ---


def test_resolve_raises_containment_error_for_parent_escape(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    with pytest.raises(WorkspaceContainmentError) as excinfo:
        tools._resolve("../escape.txt")
    assert "escapes workspace" in str(excinfo.value)
    # It is a ToolError, so invoke()'s ToolError branch also covers it.
    assert isinstance(excinfo.value, ToolError)


def test_resolve_raises_for_absolute_path_outside_root(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    with pytest.raises(WorkspaceContainmentError):
        tools._resolve(str(tmp_workspace.parent / "outside.txt"))


def test_resolve_allows_the_root_itself_and_normal_paths(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    assert tools._resolve(".") == tmp_workspace.resolve()
    assert tools._resolve("a/b/../c.txt") == (tmp_workspace / "a" / "c.txt").resolve()


async def test_parent_escape_write_is_refused_and_writes_nothing(
    tmp_workspace: Path,
) -> None:
    tools = _make_tools(tmp_workspace)
    escape_target = tmp_workspace.parent / "escape.txt"

    result = await tools.invoke(
        "write_file", {"path": "../escape.txt", "content": "pwned"}
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error is not None
    assert "escapes workspace" in result.error.lower()
    assert not escape_target.exists()


async def test_absolute_path_outside_root_is_refused_and_writes_nothing(
    tmp_workspace: Path,
) -> None:
    tools = _make_tools(tmp_workspace)
    outside = tmp_workspace.parent / "outside.txt"

    result = await tools.invoke(
        "write_file", {"path": str(outside), "content": "pwned"}
    )
    assert result.ok is False
    assert "escapes workspace" in (result.error or "").lower()
    assert not outside.exists()


async def test_symlink_escaping_the_root_is_refused_and_leaves_the_target_untouched(
    tmp_workspace: Path,
) -> None:
    tools = _make_tools(tmp_workspace)
    outside = tmp_workspace.parent / "outside.txt"
    outside.write_text("original", encoding="utf-8")
    link = tmp_workspace / "link.txt"
    link.symlink_to(outside)

    read = await tools.invoke("read_file", {"path": "link.txt"})
    assert read.ok is False
    assert "escapes workspace" in (read.error or "").lower()

    written = await tools.invoke(
        "write_file", {"path": "link.txt", "content": "pwned"}
    )
    assert written.ok is False
    assert "escapes workspace" in (written.error or "").lower()
    assert outside.read_text(encoding="utf-8") == "original"

    edited = await tools.invoke(
        "edit_file", {"path": "link.txt", "old": "original", "new": "pwned"}
    )
    assert edited.ok is False
    assert outside.read_text(encoding="utf-8") == "original"


async def test_symlinked_directory_escape_is_refused(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    secret_dir = tmp_workspace.parent / "secrets"
    secret_dir.mkdir(exist_ok=True)
    (secret_dir / "key.pem").write_text("PRIVATE KEY", encoding="utf-8")
    (tmp_workspace / "innocent").symlink_to(secret_dir, target_is_directory=True)

    result = await tools.invoke("read_file", {"path": "innocent/key.pem"})
    assert result.ok is False
    assert "escapes workspace" in (result.error or "").lower()

    listed = await tools.invoke("list_files", {"path": "innocent"})
    assert listed.ok is False
    assert "key.pem" not in (listed.output or "")


async def test_deep_parent_traversal_is_refused(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    result = await tools.invoke(
        "write_file", {"path": "../../escape.txt", "content": "pwned"}
    )
    assert result.ok is False
    assert not (tmp_workspace.parent / "escape.txt").exists()
    assert not (tmp_workspace.parent.parent / "escape.txt").exists()


# --- run_command ---


async def test_run_command_refused_when_shell_disabled(tmp_workspace: Path) -> None:
    tools = WorkspaceTools(tmp_workspace, allow_shell=False)
    result = await tools.invoke("run_command", {"command": "echo hi"})
    assert result.ok is False
    assert result.error is not None
    assert "shell access is disabled" in result.error.lower()


async def test_run_command_refused_by_default(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    result = await tools.run_command({"command": "echo hi"})
    assert result.ok is False
    assert "allow_shell" in (result.error or "")


async def test_run_command_succeeds_when_enabled(tmp_workspace: Path) -> None:
    tools = WorkspaceTools(tmp_workspace, allow_shell=True)
    result = await tools.invoke("run_command", {"command": "echo hello-from-shell"})
    assert result.ok is True
    assert "hello-from-shell" in result.output
    assert "[exit code: 0]" in result.output


async def test_run_command_runs_in_the_workspace(tmp_workspace: Path) -> None:
    tools = WorkspaceTools(tmp_workspace, allow_shell=True)
    result = await tools.invoke("run_command", {"command": "pwd"})
    assert result.ok is True
    assert str(tmp_workspace.resolve()) in result.output


async def test_run_command_reports_nonzero_exit_as_not_ok(tmp_workspace: Path) -> None:
    tools = WorkspaceTools(tmp_workspace, allow_shell=True)
    result = await tools.invoke("run_command", {"command": "exit 3"})
    assert result.ok is False
    assert "[exit code: 3]" in result.output
    assert "timed out" not in (result.error or "").lower()


async def test_run_command_missing_command_returns_tool_result(tmp_workspace: Path) -> None:
    tools = WorkspaceTools(tmp_workspace, allow_shell=True)
    result = await tools.invoke("run_command", {})
    assert result.ok is False
    assert "command must be a string" in (result.error or "")


async def test_run_command_timeout_returns_promptly_and_kills_the_child(
    tmp_workspace: Path,
) -> None:
    tools = WorkspaceTools(tmp_workspace, allow_shell=True, shell_timeout=1)
    marker = tmp_workspace / "child.pid"
    # ``exec`` keeps the same PID for the sleeping process, so the PID recorded
    # by the shell is the process asyncio spawned -- if it survives the timeout
    # the sandbox leaked a child.
    command = f"echo $$ > {shlex.quote(str(marker))}; exec sleep 30"

    started = time.monotonic()
    result = await tools.invoke("run_command", {"command": command})
    elapsed = time.monotonic() - started

    assert result.ok is False
    assert result.error is not None
    assert "timed out" in result.error.lower()
    assert "1s" in result.error
    assert elapsed < 5.0, "timeout did not return promptly"

    child_pid = int(marker.read_text(encoding="utf-8").strip())
    time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


# --- schema shapes ---


def test_specs_returns_well_formed_json_schema(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    specs = tools.specs()
    assert [spec["name"] for spec in specs] == list(tools.names())
    for spec in specs:
        assert set(spec) == {"name", "description", "input_schema"}
        assert isinstance(spec["description"], str) and spec["description"]
        schema = spec["input_schema"]
        assert schema["type"] == "object"
        assert isinstance(schema["properties"], dict) and schema["properties"]
        for prop_name, prop in schema["properties"].items():
            assert isinstance(prop_name, str)
            assert prop["type"] in {"string", "boolean", "integer", "number", "object", "array"}
        assert isinstance(schema["required"], list)
        assert set(schema["required"]) <= set(schema["properties"])
    by_name = {spec["name"]: spec for spec in specs}
    assert set(by_name["write_file"]["input_schema"]["required"]) == {"path", "content"}
    assert by_name["list_files"]["input_schema"]["required"] == []


def test_anthropic_tools_matches_the_messages_api_shape(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    anthropic_tools = tools.anthropic_tools()
    assert anthropic_tools == tools.specs()
    for tool in anthropic_tools:
        assert set(tool) == {"name", "description", "input_schema"}
        assert "type" not in tool, "Anthropic tools are flat, not nested"


def test_openai_tools_nests_parameters_under_function(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    openai_tools = tools.openai_tools()
    assert len(openai_tools) == len(tools.specs())
    for tool in openai_tools:
        assert set(tool) == {"type", "function"}
        assert tool["type"] == "function"
        assert set(tool["function"]) == {"name", "description", "parameters"}
        parameters = tool["function"]["parameters"]
        assert parameters["type"] == "object"
        assert "input_schema" not in tool["function"]
    by_name = {tool["function"]["name"]: tool for tool in openai_tools}
    assert by_name["edit_file"]["function"]["parameters"]["required"] == [
        "path",
        "old",
        "new",
    ]


# --- invoke() dispatch and argument validation ---


async def test_invoke_unknown_tool_returns_not_ok(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    result = await tools.invoke("does_not_exist", {})
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error == "Unknown tool: does_not_exist"
    assert result.output == ""


async def test_invoke_rejects_unknown_tool_with_bad_arguments(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    result = await tools.invoke("nope", {"path": "../escape.txt", "content": 5})
    assert result.ok is False
    assert "unknown tool" in (result.error or "").lower()


@pytest.mark.parametrize(
    ("tool_name", "arguments", "fragment"),
    [
        ("read_file", {}, "path must be a string"),
        ("read_file", {"path": 42}, "path must be a string"),
        ("write_file", {}, "path must be a string"),
        ("write_file", {"path": "a.txt"}, "content must be a string"),
        ("write_file", {"path": "a.txt", "content": 7}, "content must be a string"),
        ("edit_file", {"path": "a.txt", "old": 1, "new": "b"}, "old must be a string"),
        ("edit_file", {"path": "a.txt", "old": "a"}, "new must be a string"),
        (
            "edit_file",
            {"path": "a.txt", "old": "a", "new": "b", "replace_all": "yes"},
            "replace_all must be a boolean",
        ),
        ("list_files", {"path": 3}, "path must be a string"),
    ],
)
async def test_bad_arguments_return_tool_result_rather_than_raising(
    tmp_workspace: Path, tool_name: str, arguments: dict[str, Any], fragment: str
) -> None:
    tools = _make_tools(tmp_workspace)
    result = await tools.invoke(tool_name, arguments)
    assert isinstance(result, ToolResult)
    assert result.name == tool_name
    assert result.ok is False
    assert fragment in (result.error or "")


@pytest.mark.parametrize("arguments", [{}, {"command": ["ls"]}, {"command": None}])
async def test_run_command_bad_arguments_return_tool_result(
    tmp_workspace: Path, arguments: dict[str, Any]
) -> None:
    # run_command checks allow_shell first, so shell access must be on for the
    # argument validation to be reached.
    tools = WorkspaceTools(tmp_workspace, allow_shell=True)
    result = await tools.invoke("run_command", arguments)
    assert isinstance(result, ToolResult)
    assert result.name == "run_command"
    assert result.ok is False
    assert "command must be a string" in (result.error or "")


async def test_run_command_disabled_is_checked_before_arguments(tmp_workspace: Path) -> None:
    tools = WorkspaceTools(tmp_workspace, allow_shell=False)
    result = await tools.invoke("run_command", {})
    assert result.ok is False
    assert "shell access is disabled" in (result.error or "").lower()


async def test_invoke_swallows_unexpected_exceptions(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    # A path the OS layer rejects outright (embedded NUL) must still come back
    # as a ToolResult rather than propagating out of the tool loop.
    result = await tools.invoke("read_file", {"path": "\x00invalid"})
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error is not None
    assert "null byte" in result.error.lower()


async def test_write_file_reports_os_level_failures(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    result = await tools.invoke("write_file", {"path": "x" * 5000, "content": "a"})
    assert result.ok is False
    assert result.error is not None
    assert "failed to write file" in result.error.lower()


async def test_tool_result_to_dict_round_trips(tmp_workspace: Path) -> None:
    tools = _make_tools(tmp_workspace)
    result = await tools.invoke("write_file", {"path": "a.txt", "content": "a"})
    payload = result.to_dict()
    assert payload["name"] == "write_file"
    assert payload["ok"] is True
    assert payload["error"] is None
    assert "a.txt" in str(payload["output"])


async def test_list_files_output_is_truncated_past_the_entry_cap(tmp_workspace: Path) -> None:
    for index in range(520):
        (tmp_workspace / f"f{index:04d}.txt").write_text("x", encoding="utf-8")
    tools = _make_tools(tmp_workspace)
    result = await tools.invoke("list_files", {})
    assert result.ok is True
    lines = result.output.splitlines()
    assert lines[-1].startswith("... [truncated, showing first 500 of 520 entries]")
