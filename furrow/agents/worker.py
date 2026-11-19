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
    """Return the text of every ``$(...)`` and `` `...` `` substitution.

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
        if name == "patch_file":
            if "patch" not in args:
                return "error: missing patch"
            patch = args["patch"]
            if not isinstance(patch, str):
                return "error: invalid patch (expected a string)"
            return await self._patch_file(str(args.get("path", "")), patch)
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
        if len(content) > 20000:
            content = content[:20000] + "\n[... output truncated ...]"
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

    async def _patch_file(self, path: str, patch: str) -> str:
        """Apply a unified diff patch to a file."""
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            return "error: path escapes workspace"
        try:
            # Read the original file
            try:
                original = await self.client.read_file(resolved)
            except FileNotFoundError:
                return f"error: file not found: {path}"
            
            # Apply the patch
            try:
                patched = self._apply_patch(original, patch)
            except Exception as exc:
                return f"error: failed to apply patch: {exc}"
            
            # Write the patched file
            resolved.parent.mkdir(parents=True, exist_ok=True)
            await self.client.write_file(resolved, patched)
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"
        rel = str(resolved.relative_to(self.workspace.resolve()))
        self.files_changed.add(rel)
        return f"patched {path} ({len(patched.encode('utf-8'))} bytes)"

    @staticmethod
    def _apply_patch(original: str, patch: str) -> str:
        """Apply a unified diff patch to the original content."""
        import difflib
        
        original_lines = original.splitlines(keepends=True)
        patch_lines = patch.splitlines(keepends=True)
        
        # Parse the unified diff
        # Find the @@ lines to get line ranges
        hunks = []
        i = 0
        while i < len(patch_lines):
            line = patch_lines[i]
            if line.startswith('@@'):
                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                import re
                match = re.match(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@', line)
                if match:
                    old_start = int(match.group(1)) - 1  # Convert to 0-indexed
                    old_count = int(match.group(2)) if match.group(2) else 1
                    new_start = int(match.group(3)) - 1  # Convert to 0-indexed
                    new_count = int(match.group(4)) if match.group(4) else 1
                    
                    # Collect hunk lines
                    hunk_lines = []
                    i += 1
                    while i < len(patch_lines) and not patch_lines[i].startswith('@@'):
                        hunk_lines.append(patch_lines[i])
                        i += 1
                    hunks.append((old_start, old_count, new_start, new_count, hunk_lines))
                    continue
            i += 1
        
        # Apply hunks in reverse order to maintain line numbers
        result_lines = original_lines[:]
        for old_start, old_count, new_start, new_count, hunk_lines in reversed(hunks):
            # Build new lines from hunk
            new_lines = []
            for line in hunk_lines:
                if line.startswith('+') and not line.startswith('+++'):
                    new_lines.append(line[1:])
                elif line.startswith(' ') or line.startswith('-'):
                    if line.startswith('-') and not line.startswith('---'):
                        continue  # Deleted line
                    new_lines.append(line[1:] if line.startswith(' ') else line)
            
            # Replace the old lines with new lines
            end = min(old_start + old_count, len(result_lines))
            result_lines[old_start:end] = new_lines
        
        return ''.join(result_lines)

    async def _list_files(self, path: str) -> str:
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            return "error: path escapes workspace"
        try:
            entries = await asyncio.to_thread(self.client.list_files, resolved)
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"
        # Cap the listing at MAX_LIST_ENTRIES so huge trees cannot flood the context.
        rels = []
        base = self.workspace.resolve()
        for e in entries[:MAX_LIST_ENTRIES]:
            try:
                rels.append(str(Path(e).relative_to(base)) if Path(e).is_absolute() else e)
            except ValueError:
                rels.append(str(e))
        return "\n".join(rels)

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
