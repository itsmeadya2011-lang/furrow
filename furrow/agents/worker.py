from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shlex
import signal
from pathlib import Path
from typing import Any

from furrow.agents.prompts import WORKER_SYSTEM, WORKER_TOOLS
from furrow.config import Settings, TaskModel, WorkerResult
from furrow.config import settings as global_settings
from furrow.llm import LLMClient, LLMResponse, ToolCall

# Substring patterns that are denied in a command. They are matched
# case-insensitively against the normalized command.
DENIED_COMMANDS = (
    ":(){:|:&};:",
    "mkfs",
    "dd if=",
    "shutdown",
    "reboot",
    "> /dev/sda",
)

# Programs that are denied by name: the basename of the first token of a
# segment. ``mkfs`` variants (``mkfs.ext4`` and friends) are denied as well.
DENIED_PROGRAMS = frozenset(
    {
        "dd",
        "fdisk",
        "halt",
        "init",
        "mkfs",
        "parted",
        "poweroff",
        "reboot",
        "shutdown",
    }
)

# Stop reasons that indicate the model response was truncated, so a worker
# turn with no tool calls is a failure rather than a completed task.
TRUNCATED_STOP_REASONS = frozenset({"max_tokens", "length"})

# Maximum number of entries returned by the ``list_files`` tool.
MAX_LIST_ENTRIES = 500

# Maximum number of characters returned by the ``read_file`` tool.
MAX_READ_CHARS = 20000

# Shell separators a command is split on. A lone ``&`` backgrounds a command,
# so the command on each side is checked as its own segment.
_SEGMENT_SPLIT_RE = re.compile(r"(?:&&|\|\||;|&|(?<!>)\||\n|\r)+")

# Every spelling of the HOME variable. The look-ahead keeps "$HOME_DIR" and
# friends untouched.
_HOME_VAR = r"\$\{?home\}?(?![A-Za-z0-9_])"
_HOME_VAR_RE = re.compile(
    rf'"(?:{_HOME_VAR})"' rf"|'(?:{_HOME_VAR})'" rf"|{_HOME_VAR}",
    re.IGNORECASE,
)

# The target of a redirection: ">> file", ">| file" (clobber override) or
# "> file". ``&>`` and ``>&`` are caught because their ``>`` is matched here.
_REDIRECT_RE = re.compile(r"(?:>>|>\|?)\s*([^\s;|&<>]+)")

# How deep command substitutions are followed: a substitution may contain
# another substitution.
_MAX_SUBSTITUTION_DEPTH = 3

# Leading program names that are only wrappers: they run the command that
# follows them, so a dangerous ``rm``/``chmod`` hides behind them.
_WRAPPER_PROGRAMS = frozenset({"sudo", "doas", "command", "env", "nohup", "exec"})

# Regex for a leading variable assignment such as ``FOO=1`` or ``PATH=/bin``.
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Targets that discard output rather than write a real file.
_SPECIAL_WRITE_TARGETS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})


def _normalize_command(command: str) -> str:
    """Normalize a command so that it can be matched.

    The command is lower-cased, runs of whitespace are collapsed into a
    single space, the quotes around the whole command are removed and every
    spelling of the HOME variable (``$HOME``, ``${HOME}``, ``"$HOME"``,
    ``'$HOME'``, ``$home``, ``${home}``) is canonicalised to the single
    token ``$home``.
    """
    if not command:
        return ""
    cmd = command.strip()
    # Strip surrounding quotes (single/double) of whole command
    while len(cmd) >= 2 and ((cmd[0] == cmd[-1]) and cmd[0] in ('"', "'")):
        cmd = cmd[1:-1].strip()
    cmd = cmd.lower()
    # Canonicalize every spelling of the HOME variable to "$home".
    cmd = _HOME_VAR_RE.sub("$home", cmd)
    # Collapse runs of whitespace to single space
    cmd = re.sub(r'\s+', ' ', cmd)
    return cmd


def _split_command_segments(command: str) -> list[str]:
    """Split a command into segments on the shell separators ``;``, ``&&``,
    ``||``, ``|``, ``&`` (background) and newlines."""
    if not command:
        return []
    segments = _SEGMENT_SPLIT_RE.split(command)
    return [s for s in segments if s.strip()]  # Remove empty segments


def _shell_tokens(segment: str) -> list[str]:
    """Split a segment into shell tokens.

    A segment that cannot be parsed (unbalanced quotes, for example) falls
    back to a plain whitespace split.
    """
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _get_program_basename(segment: str) -> str | None:
    """Get basename of first token in segment."""
    tokens = _shell_tokens(segment.strip())
    if not tokens:
        return None
    return os.path.basename(tokens[0])


def _program_and_args(segment: str) -> tuple[str, list[str]] | None:
    """Split a segment into the effective program and its argument tokens.

    Leading variable assignments (``FOO=1``) and wrapper program names
    (``sudo``, ``env``, ``nohup``, ...) are skipped so that the real program
    that runs is reported, together with the remaining argument tokens.
    Wrapper option flags are skipped too.
    """
    tokens = _shell_tokens(segment.strip())
    if not tokens:
        return None
    VALUE_OPTS = {
        "sudo": {"-u", "--user", "-g", "--group", "-p", "--prompt"},
        "doas": {"-u", "--user", "-g", "--group", "-p", "--prompt"},
        "env": {"-u", "--unset", "-C", "--chdir"},
    }
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if _ASSIGNMENT_RE.match(token):
            i += 1
            continue
        base = os.path.basename(token).lower()
        if base in _WRAPPER_PROGRAMS:
            i += 1
            value_opts = VALUE_OPTS.get(base, set())
            while i < len(tokens):
                t = tokens[i]
                if t == "--":
                    i += 1
                    break
                if base == "env" and _ASSIGNMENT_RE.match(t):
                    i += 1
                    continue
                if t.startswith("--"):
                    if "=" in t:
                        i += 1
                        continue
                    if t in value_opts:
                        i += 2
                        continue
                    i += 1
                    continue
                if t.startswith("-"):
                    if t in value_opts:
                        i += 2
                        continue
                    i += 1
                    continue
                break
            continue
        break
    if i >= len(tokens):
        return None
    return os.path.basename(tokens[i]).lower(), tokens[i + 1 :]


def _is_short_flag_cluster(token: str) -> bool:
    """Report whether a token is a short-flag cluster such as ``-rf``.

    A cluster starts with a single ``-`` (not ``--``) and is followed by at
    least one letter.
    """
    if len(token) < 2 or not token.startswith("-") or token.startswith("--"):
        return False
    return any(ch.isalpha() for ch in token[1:])


def _short_cluster_has(token: str, letters: str) -> bool:
    """Report whether a short-flag cluster contains any of the given letters."""
    if not _is_short_flag_cluster(token):
        return False
    return any(ch in letters for ch in token[1:])


def _is_dangerous_rm(args: list[str], workspace: Path) -> bool:
    """Report whether an ``rm`` invocation recursively forces outside the workspace.

    Only a command that is both recursive and forced is dangerous. Each target
    is denied if it contains a shell expansion (``$`` or a backtick) or resolves
    outside of the workspace.
    """
    recursive = False
    force = False
    targets: list[str] = []
    options_done = False
    for token in args:
        if options_done:
            targets.append(token)
            continue
        if token == "--":
            options_done = True
            continue
        if token == "--recursive":
            recursive = True
            continue
        if token == "--force":
            force = True
            continue
        if token.startswith("--"):
            continue
        if token.startswith("-"):
            if _short_cluster_has(token, "rR"):
                recursive = True
            if _short_cluster_has(token, "f"):
                force = True
            continue
        targets.append(token)
    if recursive and force:
        for target in targets:
            if "$" in target or "`" in target:
                return True
            if not _target_inside_workspace(target, workspace):
                return True
    return False


def _is_dangerous_chmod(args: list[str], workspace: Path) -> bool:
    """Report whether a recursive ``chmod`` targets outside the workspace."""
    recursive = False
    targets: list[str] = []
    options_done = False
    for token in args:
        if options_done:
            targets.append(token)
            continue
        if token == "--":
            options_done = True
            continue
        if token == "--recursive":
            recursive = True
            continue
        if token.startswith("--"):
            continue
        if token.startswith("-"):
            if _short_cluster_has(token, "rR"):
                recursive = True
            continue
        targets.append(token)
    if recursive:
        for target in targets:
            if "$" in target or "`" in target:
                return True
            if not _target_inside_workspace(target, workspace):
                return True
    return False


def _extract_command_substitutions(command: str) -> list[str]:
    """Return the text of every ``$(...)``, `` `...` `` and ``<(...)/>(...)`` substitution.

    The shell runs the substituted text, so it has to be checked with the
    very same rules as the command itself.
    """
    substitutions: list[str] = []
    index = 0
    length = len(command)
    while index < length:
        char = command[index]
        if char == "`":
            end = command.find("`", index + 1)
            if end == -1:
                break
            text = command[index + 1 : end]
            if text.strip():
                substitutions.append(text)
            index = end + 1
            continue
        if char == "$" and command[index + 1 : index + 2] == "(":
            # Walk to the matching closing parenthesis, nested ones included.
            depth = 0
            cursor = index + 1
            while cursor < length:
                if command[cursor] == "(":
                    depth += 1
                elif command[cursor] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                cursor += 1
            if cursor >= length:
                break  # unbalanced parentheses
            text = command[index + 2 : cursor]
            if text.strip():
                substitutions.append(text)
            index = cursor + 1
            continue
        if char in "<>" and command[index + 1 : index + 2] == "(":
            # Walk to the matching closing parenthesis of a process
            # substitution <(...) or >(...), nested ones included. The
            # `(` of `$(...)` is handled by the `$(` branch above, so it is
            # never mistaken for a process substitution here.
            depth = 0
            cursor = index + 1
            while cursor < length:
                if command[cursor] == "(":
                    depth += 1
                elif command[cursor] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                cursor += 1
            if cursor >= length:
                break  # unbalanced parentheses
            text = command[index + 2 : cursor]
            if text.strip():
                substitutions.append(text)
            index = cursor + 1
            continue
        index += 1
    return substitutions


def _expand_home(target: str) -> str:
    """Expand ``~`` and the HOME variable of a redirection target."""
    home = os.path.expanduser("~")
    return os.path.expanduser(_HOME_VAR_RE.sub(home, target))


def _target_inside_workspace(target: str, workspace: Path) -> bool:
    """Resolve a redirection target and report whether it stays in the workspace.

    An absolute target is resolved as it is, a relative one against the
    workspace.
    """
    base = workspace.resolve()
    path = Path(_expand_home(target))
    resolved = path if path.is_absolute() else base / path
    try:
        resolved.resolve().relative_to(base)
    except ValueError:
        return False
    return True


def _redirects_outside_workspace(segment: str, workspace: Path) -> bool:
    """Report whether a segment redirects output outside of the workspace."""
    for match in _REDIRECT_RE.finditer(segment):
        target = match.group(1).strip().strip("\"'")
        if not target:
            continue
        # /dev/null, /dev/stdout and /dev/stderr discard output rather than
        # write a real file, so they are never a policy violation.
        if target in _SPECIAL_WRITE_TARGETS:
            continue
        # A target that still contains a shell expansion cannot be resolved
        # statically: the shell expands it before opening the file.
        if "$" in target or "`" in target:
            return True
        if not _target_inside_workspace(target, workspace):
            return True
    return False


# Programs that write to path arguments rather than via shell redirections.
_WRITE_PATH_PROGRAMS = frozenset({"tee"})


def _git_subcommand(args: list[str]) -> str | None:
    """Return the git subcommand, skipping global options that take a value."""
    VALUE_OPTS = {"-c", "-C", "--git-dir", "--work-tree", "--namespace",
                  "--exec-path"}
    i = 0
    while i < len(args):
        t = args[i]
        if t == "--":
            i += 1
            continue
        if t.startswith("--"):
            if "=" in t:
                i += 1
                continue
            if t in VALUE_OPTS:
                i += 2
                continue
            i += 1
            continue
        if t.startswith("-"):
            if t in VALUE_OPTS or t in ("-C", "-c"):
                i += 2
                continue
            i += 1
            continue
        return t
    return None


def _is_force_push(segment: str, normalized: str) -> bool:
    """Report whether a segment is a git force push."""
    parsed = _program_and_args(segment)
    if parsed is None:
        return False
    name, args = parsed
    if name != "git":
        return False
    sub = _git_subcommand(args)
    if sub != "push":
        return False
    push_args = args
    # find push args after subcommand? use all args; flags may appear anywhere
    # Covers --force, --force-with-lease and --force-with-lease=<ref>.
    if any(a.lower().startswith("--force") for a in push_args):
        return True
    if any(a.lower() == "-f" for a in push_args):
        return True
    # A "+" refspec forces the update on the remote
    return any(a.startswith("+") or ":+" in a for a in push_args)


def _writes_outside_workspace(segment: str, workspace: Path) -> bool:
    """Report whether a segment writes to a path argument outside the workspace.

    This catches programs that write files through arguments rather than shell
    redirections (e.g. ``tee``). Non-option tokens are treated as path
    arguments; a path outside the workspace or one that still contains a shell
    expansion (``$`` or a backtick) is denied. A program with no path argument
    is allowed.
    """
    parsed = _program_and_args(segment)
    if parsed is None:
        return False
    name, args = parsed
    if name not in _WRITE_PATH_PROGRAMS:
        return False
    for token in args:
        if token.startswith("-"):
            continue
        target = token.strip().strip("\"'")
        if not target:
            continue
        if target in _SPECIAL_WRITE_TARGETS:
            continue
        if "$" in target or "`" in target:
            return True
        if not _target_inside_workspace(target, workspace):
            return True
    return False


def _is_dangerous_segment(segment: str, workspace: Path) -> bool:
    """Report whether a single command segment violates the command policy."""
    normalized = _normalize_command(segment)
    if not normalized:
        return False

    # DENY by PROGRAM NAME (first token)
    program = _get_program_basename(segment)
    if program:
        program = program.lower()
        if program in DENIED_PROGRAMS or program.startswith("mkfs."):
            return True

    # DENY by PATTERN
    if any(denied.lower() in normalized for denied in DENIED_COMMANDS):
        return True

    # DENY dangerous rm/chmod (token-aware, not substring-based). Works even
    # behind wrapper programs such as sudo/env and inside substitutions.
    # Also check the resolved program name against DENIED_PROGRAMS and mkfs.*
    parsed = _program_and_args(segment)
    if parsed is not None:
        name, args = parsed
        if name in DENIED_PROGRAMS or name.startswith("mkfs."):
            return True
        if name == "rm" and _is_dangerous_rm(args, workspace):
            return True
        if name == "chmod" and _is_dangerous_chmod(args, workspace):
            return True

    # DENY a force push
    if _is_force_push(segment, normalized):
        return True

    # DENY a redirection or a file-writing argument that escapes the workspace
    if _redirects_outside_workspace(segment, workspace):
        return True
    return bool(_writes_outside_workspace(segment, workspace))


def _text_is_denied(text: str, workspace: Path, depth: int) -> bool:
    """Check a command - or the text of a command substitution - against the policy."""
    normalized = _normalize_command(text)
    # The whole text is checked against the pattern list as well, so that a
    # denied pattern cannot hide in an unusual position (inside quotes, in
    # the middle of a pipeline, ...).
    if any(denied.lower() in normalized for denied in DENIED_COMMANDS):
        return True
    # Every segment of the text is checked on its own.
    if any(_is_dangerous_segment(segment, workspace) for segment in _split_command_segments(text)):
        return True
    # The shell runs the substituted text, so it is checked as well.
    if depth < _MAX_SUBSTITUTION_DEPTH:
        for substitution in _extract_command_substitutions(text):
            if _text_is_denied(substitution, workspace, depth + 1):
                return True
    return False


def is_command_denied(command: str, workspace: Path) -> bool:
    """Check if command is denied by security policy."""
    if not command or not command.strip():
        return False
    return _text_is_denied(command, workspace, depth=0)


def resolve_in_workspace(workspace: Path, path: str) -> Path | None:
    """Return the absolute path for `path` inside `workspace`, or None when it escapes."""
    base = Path(workspace).resolve()
    if not path:
        return None
    p = Path(path)
    candidate = (p if p.is_absolute() else base / p).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
        goal: str = "",
    ) -> None:
        self.task = task
        self.settings = settings if settings is not None else global_settings
        self.workspace = Path(workspace) if workspace is not None else Path(self.settings.workspace)
        self.client = client if client is not None else LLMClient(settings=self.settings)
        self.goal = goal
        self.files_changed: set[str] = set()

    async def run(self) -> WorkerResult:
        try:
            return await self._run()
        except Exception as exc:  # pragma: no cover - defensive
            return WorkerResult(summary=str(exc), success=False, error=str(exc), iterations=0)

    async def _run(self) -> WorkerResult:
        max_iters = self.settings.max_worker_iterations
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": self._initial_text(),
            }
        ]
        iterations = 0
        for i in range(max_iters):
            iterations = i + 1
            response: LLMResponse = await self.client.chat(
                messages, tools=WORKER_TOOLS, system=WORKER_SYSTEM, model=self.settings.worker_model
            )
            if not response.tool_calls:
                summary = response.text.strip() if response.text.strip() else "Task completed."
                if response.stop_reason in TRUNCATED_STOP_REASONS:
                    return WorkerResult(
                        summary=summary,
                        success=False,
                        files_changed=sorted(self.files_changed),
                        iterations=iterations,
                        error=f"worker response truncated (stop_reason={response.stop_reason})",
                    )
                return WorkerResult(
                    summary=summary,
                    success=True,
                    files_changed=sorted(self.files_changed),
                    iterations=iterations,
                )
            # append assistant turn
            blocks: list[dict[str, Any]] = []
            if response.text:
                blocks.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            messages.append({"role": "assistant", "content": blocks})

            # execute tools
            result_blocks: list[dict[str, Any]] = []
            for tc in response.tool_calls:
                result = await self._execute_tool(tc)
                result_blocks.append(
                    {"type": "tool_result", "tool_use_id": tc.id, "content": result}
                )
            messages.append({"role": "user", "content": result_blocks})

        # budget exhausted: the worker was truncated, so this is NOT a success.
        # Reporting success here would make the orchestrator mark the task as
        # completed instead of retrying it.
        return WorkerResult(
            summary="worker iteration budget exhausted",
            success=False,
            files_changed=sorted(self.files_changed),
            iterations=max_iters,
            error="worker iteration budget exhausted",
        )

    def _initial_text(self) -> str:
        parts = []
        if self.goal:
            parts.append(f"Overall goal: {self.goal}")
        parts.append(f"Task: {self.task.description}")
        parts.append(f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}")
        parts.append(f"Workspace: {self.workspace}")
        return "\n".join(parts)

    async def _execute_tool(self, tc: ToolCall) -> str:
        name = tc.name
        args = self._coerce_tool_args(tc.arguments)
        if args is None:
            return "error: invalid tool arguments"
        if name == "read_file":
            return await self._read_file(str(args.get("path", "")))
        if name == "write_file":
            if "content" not in args:
                return "error: missing content"
            content = args["content"]
            if not isinstance(content, str):
                return "error: invalid content (expected a string)"
            return await self._write_file(str(args.get("path", "")), content)
        if name == "edit_file":
            if "edits" not in args:
                return "error: missing edits"
            edits = args["edits"]
            if not isinstance(edits, list) or not edits:
                return "error: edits must be a non-empty list"
            return await self._edit_file(str(args.get("path", "")), edits)
        if name == "search_files":
            if "pattern" not in args:
                return "error: missing pattern"
            return await self._search_files(
                str(args.get("pattern", "")),
                str(args.get("path", "")) if args.get("path") else ".",
                str(args.get("include", "")) if args.get("include") else None,
                int(args.get("max_results", 50)) if args.get("max_results") else 50,
            )
        if name == "patch_file":
            if "diff" not in args:
                return "error: missing diff"
            return await self._patch_file(
                str(args.get("path", "")),
                str(args.get("diff", "")),
            )
        if name == "list_files":
            return await self._list_files(
                str(args.get("path", "")) if args.get("path") is not None else "."
            )
        if name == "run_command":
            return await self._run_command(str(args.get("command", "")))
        return f"unknown tool: {name}"

    @staticmethod
    def _coerce_tool_args(arguments: Any) -> dict[str, Any] | None:
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            text = arguments.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except (ValueError, RecursionError):
                return None
            if isinstance(parsed, dict):
                return parsed
            return None
        return None

    async def _read_file(self, path: str) -> str:
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            return "error: path escapes workspace"
        try:
            content = await self.client.read_file(resolved)
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"
        if len(content) > MAX_READ_CHARS:
            content = (
                content[:MAX_READ_CHARS]
                + f"\n[... output truncated ...] ({len(content)} chars total)"
            )
        return content

    async def _write_file(self, path: str, content: str) -> str:
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            return "error: path escapes workspace"
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            await self.client.write_file(resolved, content)
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"
        rel = str(resolved.relative_to(self.workspace.resolve()))
        self.files_changed.add(rel)
        return f"wrote {path} ({len(content.encode('utf-8'))} bytes)"

    async def _edit_file(
        self,
        path: str,
        edits: list[dict[str, Any]] | None = None,
        diff: str | None = None,
    ) -> str:
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            return "error: path escapes workspace"
        try:
            content = await self.client.read_file(resolved)
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"

        replacements: list[tuple[str, str]] = []
        if diff is not None:
            try:
                new_content = _apply_unified_diff(content, diff)
            except ValueError as exc:
                return f"error: {exc}"
        elif edits is not None:
            for index, edit in enumerate(edits):
                if not isinstance(edit, dict):
                    return f"error: edit #{index} is not an object"
                old_text = edit.get("old_text", "")
                new_text = edit.get("new_text", "")
                if not isinstance(old_text, str) or not isinstance(new_text, str):
                    return f"error: edit #{index} old_text/new_text must be strings"
                replacements.append((old_text, new_text))
            new_content = content
            for old_text, new_text in replacements:
                count = new_content.count(old_text)
                if count == 0:
                    return f"error: old_text not found in {path}"
                if count > 1:
                    return (
                        f"error: old_text appears {count} times in {path}; "
                        f"edit is ambiguous"
                    )
                new_content = new_content.replace(old_text, new_text, 1)
        else:
            return "error: missing edits or diff"

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            await self.client.write_file(resolved, new_content)
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"

        rel = str(resolved.relative_to(self.workspace.resolve()))
        self.files_changed.add(rel)
        return f"edited {path}"


def _apply_unified_diff(content: str, diff: str) -> str:
    """Apply a unified diff to `content`, returning the new content.

    Raises ValueError if the diff cannot be parsed or applied.
    Hunks are applied in reverse order so earlier line numbers stay valid.
    """
    original_lines = content.splitlines(keepends=True)
    diff_lines = diff.splitlines()

    # Locate the first hunk header (@@ ... @@).
    hunk_index = 0
    while hunk_index < len(diff_lines) and not diff_lines[hunk_index].startswith("@@"):
        hunk_index += 1
    if hunk_index >= len(diff_lines):
        raise ValueError("no hunks found in diff")

    hunks = []
    cursor = hunk_index
    while cursor < len(diff_lines):
        line = diff_lines[cursor]
        if not line.startswith("@@"):
            break
        header = line
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)", header)
        if not m:
            raise ValueError(f"malformed hunk header: {header}")
        old_start = int(m.group(1))
        old_count = int(m.group(2) or 1)
        new_start = int(m.group(3))
        new_count = int(m.group(4) or 1)
        cursor += 1

        ops: list[tuple[str, str]] = []  # ("context"/"remove"/"add", text)
        removed = 0
        added = 0
        while cursor < len(diff_lines):
            dl = diff_lines[cursor]
            if dl.startswith("@@"):
                break
            if dl.startswith("---") or dl.startswith("+++"):
                cursor += 1
                continue
            if dl.startswith("+"):
                ops.append(("add", dl[1:]))
                added += 1
            elif dl.startswith("-"):
                ops.append(("remove", dl[1:]))
                removed += 1
            elif dl.startswith(" "):
                ops.append(("context", dl[1:]))
                removed += 1
                added += 1
            elif dl == "":
                # Blank context line (a line that was empty becomes "" with no
                # leading space in some diff emitters). Treat as context.
                ops.append(("context", ""))
                removed += 1
                added += 1
            else:
                # Unexpected line: end of this hunk.
                break
            cursor += 1

        if removed != old_count or added != new_count:
            raise ValueError(
                f"hunk counts mismatch: expected -{old_count}/+{new_count}, "
                f"got -{removed}/+{added}"
            )
        hunks.append((old_start, old_count, ops))

    # Apply hunks in reverse order so line numbers of earlier hunks are stable.
    lines = list(original_lines)
    for old_start, old_count, ops in reversed(hunks):
        start = old_start - 1  # 1-based to 0-based
        if start < 0 or start + old_count > len(lines):
            raise ValueError(f"hunk at line {old_start} is out of range")
        # Verify the removed lines match the original context.
        segment = lines[start : start + old_count]
        for i, (kind, text) in enumerate(ops):
            if kind in ("remove", "context"):
                expected = segment[i] if i < len(segment) else ""
                if expected.rstrip("\n") != text:
                    raise ValueError(
                        f"context mismatch at line {old_start + i}: "
                        f"expected {expected!r}, got {text!r}"
                    )
        replacement = [text + "\n" for kind, text in ops if kind in ("context", "add")]
        lines[start : start + old_count] = replacement

    return "".join(lines)

    async def _edit_file(self, path: str, edits: list[dict]) -> str:
        """Apply a list of search/replace edits to a file atomically.

        Each edit is a dict with 'old_text' and 'new_text'. Empty old_text means
        insert at the start of the file; empty new_text means delete.
        """
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            return "error: path escapes workspace"

        # Validate edits structure
        for i, edit in enumerate(edits):
            if not isinstance(edit, dict):
                return f"error: edit {i} must be an object"
            if "old_text" not in edit or "new_text" not in edit:
                return f"error: edit {i} must have 'old_text' and 'new_text'"
            if not isinstance(edit["old_text"], str) or not isinstance(edit["new_text"], str):
                return f"error: edit {i} old_text and new_text must be strings"

        try:
            # Read current content
            try:
                content = await self.client.read_file(resolved)
            except FileNotFoundError:
                return f"error: file not found: {path}"
            except (OSError, ValueError, RuntimeError) as exc:
                return f"error: {exc}"

            original_content = content
            for i, edit in enumerate(edits):
                old_text = edit["old_text"]
                new_text = edit["new_text"]

                if old_text == "":
                    # Insert at the beginning of the file
                    content = new_text + content
                else:
                    # Find the old_text
                    if old_text not in content:
                        return f"error: edit {i}: old_text not found in file"
                    # Replace only the first occurrence
                    content = content.replace(old_text, new_text, 1)

            # Write the modified content
            try:
                await self.client.write_file(resolved, content)
            except (OSError, ValueError, RuntimeError) as exc:
                return f"error: {exc}"

            rel = str(resolved.relative_to(self.workspace.resolve()))
            self.files_changed.add(rel)
            return f"edited {path} ({len(edits)} edit(s) applied)"

        except Exception as exc:
            return f"error: {exc}"

    async def _list_files(self, path: str) -> str:
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            return "error: path escapes workspace"
        try:
            entries = await asyncio.to_thread(self.client.list_files, resolved)
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"
        # Cap the listing at MAX_LIST_ENTRIES so huge trees cannot flood the
        # context, and tell the model when entries were left out so it knows
        # to list a subdirectory instead of assuming it saw everything.
        total = len(entries)
        rels = []
        base = self.workspace.resolve()
        for e in entries[:MAX_LIST_ENTRIES]:
            try:
                rels.append(str(Path(e).relative_to(base)) if Path(e).is_absolute() else e)
            except ValueError:
                rels.append(str(e))
        text = "\n".join(rels)
        omitted = total - len(rels)
        if omitted > 0:
            note = f"[... {omitted} more entries not shown; list a subdirectory ...]"
            text = f"{text}\n{note}" if text else note
        return text

    async def _search_files(
        self, pattern: str, path: str, include: str | None, max_results: int
    ) -> str:
        """Search for a regex pattern in files under the workspace."""
        search_path = resolve_in_workspace(self.workspace, path)
        if search_path is None:
            return "error: path escapes workspace"

        try:
            import re as regex_module

            regex = regex_module.compile(pattern)
        except regex_module.error as exc:
            return f"error: invalid regex pattern: {exc}"

        matches: list[str] = []
        base = self.workspace.resolve()

        try:
            # Get all files to search
            all_files = await asyncio.to_thread(self.client.list_files, search_path)

            # Filter by include pattern if provided
            if include:
                import fnmatch

                all_files = [f for f in all_files if fnmatch.fnmatch(f, include)]

            for file_path in all_files:
                if len(matches) >= max_results:
                    break

                # Resolve full path
                full_path = (search_path / file_path) if not Path(file_path).is_absolute() else Path(file_path)
                try:
                    full_path = full_path.resolve()
                    full_path.relative_to(base)
                except ValueError:
                    continue  # Skip files outside workspace

                # Skip directories
                if not full_path.is_file():
                    continue

                # Skip binary/large files
                try:
                    stat = full_path.stat()
                    if stat.st_size > 1_000_000:  # 1MB limit
                        continue
                except OSError:
                    continue

                # Read and search file
                try:
                    content = await self.client.read_file(full_path)
                except (OSError, ValueError, RuntimeError):
                    continue

                for line_num, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        rel_path = str(full_path.relative_to(base))
                        # Show context: the matching line with line number
                        matches.append(f"{rel_path}:{line_num}: {line.strip()}")
                        if len(matches) >= max_results:
                            break

        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"

        if not matches:
            return f"No matches found for pattern: {pattern}"

        return "\n".join(matches)

    async def _patch_file(self, path: str, diff: str) -> str:
        """Apply a unified diff to a file."""
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            return "error: path escapes workspace"

        try:
            # Read current content
            try:
                content = await self.client.read_file(resolved)
            except FileNotFoundError:
                return f"error: file not found: {path}"
            except (OSError, ValueError, RuntimeError) as exc:
                return f"error: {exc}"

            # Apply the diff
            try:
                new_content = _apply_unified_diff(content, diff)
            except ValueError as exc:
                return f"error: failed to apply diff: {exc}"

            # Write the modified content
            try:
                await self.client.write_file(resolved, new_content)
            except (OSError, ValueError, RuntimeError) as exc:
                return f"error: {exc}"

            rel = str(resolved.relative_to(self.workspace.resolve()))
            self.files_changed.add(rel)
            return f"patched {path}"

        except Exception as exc:
            return f"error: {exc}"

    async def _run_command(self, command: str) -> str:
        if not command.strip():
            return "error: empty command"
        # The hardened check subsumes the legacy pattern list: it normalizes
        # the command, checks every segment against DENIED_COMMANDS and
        # DENIED_PROGRAMS, and denies force pushes and redirections outside
        # the workspace.
        if is_command_denied(command, self.workspace):
            return "error: command denied"
        try:
            shlex.split(command)
        except ValueError:
            return "error: could not parse command"
        timeout = float(getattr(self.settings, "command_timeout", 300.0) or 300.0)
        # On POSIX the command runs in its own session/process group so that a
        # timeout can kill the whole tree (children and grandchildren included)
        # instead of leaking orphans. On other platforms we fall back to
        # killing the direct child only.
        new_session = os.name == "posix"
        spawn_kwargs: dict[str, Any] = {"start_new_session": True} if new_session else {}
        try:
            if os.name == "posix":
                proc = await asyncio.create_subprocess_exec(
                    "/bin/sh", "-c", command,
                    cwd=str(self.workspace),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **spawn_kwargs,
                )
            else:
                parts = shlex.split(command)
                if not parts:
                    return "error: empty command"
                proc = await asyncio.create_subprocess_exec(
                    *parts,
                    cwd=str(self.workspace),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **spawn_kwargs,
                )
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            await self._kill_process_tree(proc, new_session)
            return "command timed out"
        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        combined = out + "\n[stderr]\n" + err if err.strip() else out
        cap = 20000
        if len(combined) > cap:
            combined = "[... output truncated ...]\n" + combined[-cap:]
        if proc.returncode is not None and proc.returncode != 0:
            combined += f"\n[exit status {proc.returncode}]"
        return combined

    @staticmethod
    async def _kill_process_tree(proc: asyncio.subprocess.Process, new_session: bool) -> None:
        """Kill a timed-out command together with every process it spawned."""
        if new_session:
            try:
                pgid = os.getpgid(proc.pid)
            except (OSError, AttributeError):
                pgid = None
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (OSError, AttributeError):
                    pass
                else:
                    with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                    return
        # Fallback: no process group available (or killing it failed).
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.kill()
