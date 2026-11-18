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
    """
    tokens = _shell_tokens(segment.strip())
    if not tokens:
        return None
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if _ASSIGNMENT_RE.match(token):
            i += 1
            continue
        base = os.path.basename(token).lower()
        if base in _WRAPPER_PROGRAMS:
            i += 1
            # ``env`` can take a variable assignment right after it.
            if base == "env" and i < len(tokens) and _ASSIGNMENT_RE.match(tokens[i]):
                i += 1
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
    """Report whether an ``rm`` invocation targets outside the workspace.

    Every target is denied when it contains a shell expansion (``$`` or a
    backtick) or resolves outside of the workspace, regardless of the
    ``-r``/``-f`` flags. Flag parsing (including ``--``) is still honoured.
    """
    targets: list[str] = []
    options_done = False
    for token in args:
        if options_done:
            targets.append(token)
            continue
        if token == "--":
            options_done = True
            continue
        if token.startswith("--"):
            continue
        if token.startswith("-"):
            continue
        targets.append(token)
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
# ``cp``/``mv``/``install``/``ln``/``rsync`` write to their last non-option
# operand (or to the ``-t``/``--target-directory`` directory); the others
# treat every non-option operand as a write target.
_WRITE_PATH_PROGRAMS = frozenset(
    {"tee", "cp", "mv", "install", "ln", "rsync", "truncate", "touch", "mkdir", "rmdir"}
)
_LAST_DEST_PROGRAMS = frozenset({"cp", "mv", "install", "ln", "rsync"})


def _is_force_push(segment: str, normalized: str) -> bool:
    """Report whether a segment is a git force push."""
    if "git" not in normalized or "push" not in normalized:
        return False
    # Covers --force, --force-with-lease and --force-with-lease=<ref>.
    if "--force" in normalized or "--force-with-lease" in normalized:
        return True
    tokens = _shell_tokens(segment)
    # A short -f flag only counts as a shell token, so that ordinary
    # commands (e.g. "ls -f") are not mistaken for a force push.
    if any(token.lower() == "-f" for token in tokens):
        return True
    # A "+" refspec forces the update on the remote (e.g. "git push origin
    # +main" or "git push origin HEAD:+main").
    return any(token.startswith("+") or ":+" in token for token in tokens)


def _writes_outside_workspace(segment: str, workspace: Path) -> bool:
    """Report whether a segment writes to a path argument outside the workspace.

    This catches programs that write files through arguments rather than shell
    redirections (e.g. ``tee``, ``cp``, ``mv``, ...). Option tokens (and their
    values for ``-t``/``--target-directory``) are skipped; ``--`` terminates
    option parsing. For ``cp``/``mv``/``install``/``ln``/``rsync`` only the
    destination (last operand, or the target directory) is checked; for the
    others every non-option operand is a write target. A path outside the
    workspace or one that still contains a shell expansion (``$`` or a
    backtick) is denied. A program with no path argument is allowed.
    """
    parsed = _program_and_args(segment)
    if parsed is None:
        return False
    name, args = parsed
    if name not in _WRITE_PATH_PROGRAMS:
        return False
    options_done = False
    operands: list[str] = []
    target_dir: str | None = None
    i = 0
    while i < len(args):
        token = args[i]
        if options_done:
            operands.append(token)
            i += 1
            continue
        if token == "--":
            options_done = True
            i += 1
            continue
        if token.startswith("--target-directory="):
            target_dir = token.split("=", 1)[1]
            i += 1
            continue
        if token == "--target-directory":
            if i + 1 < len(args):
                target_dir = args[i + 1]
                i += 2
                continue
            i += 1
            continue
        if token == "-t":
            if i + 1 < len(args):
                target_dir = args[i + 1]
                i += 2
                continue
            i += 1
            continue
        if token.startswith("-t") and len(token) > 2:
            target_dir = token[2:]
            i += 1
            continue
        if token.startswith("-"):
            i += 1
            continue
        operands.append(token)
        i += 1

    targets: list[str] = []
    if target_dir is not None:
        targets.append(target_dir)
    if name in _LAST_DEST_PROGRAMS:
        if operands:
            targets.append(operands[-1])
    else:
        targets.extend(operands)

    for target in targets:
        target = target.strip().strip("\"'")
        if not target:
            continue
        if target in _SPECIAL_WRITE_TARGETS:
            continue
        if "$" in target or "`" in target:
            return True
        if not _target_inside_workspace(target, workspace):
            return True
    return False


# Destructive ``find`` actions that run a command or remove matches.
_FIND_ACTION_TOKENS = frozenset({"-delete", "-exec", "-execdir", "-ok", "-okdir"})

# Global ``find`` options that may precede the roots (``find -L /etc ...``).
_FIND_GLOBAL_FLAGS = frozenset({"-H", "-L", "-P"})

# ``perl`` options that consume the rest of the token as their value, so an
# ``e`` inside that value is not a ``-e`` script option.
_PERL_VALUE_REST_OPTIONS = frozenset({"M", "m", "I", "d", "D", "0"})

# ``xargs`` options that consume a separate value and options that do not.
_XARGS_VALUE_OPTIONS = frozenset({"-n", "-L", "-I", "-P", "-s", "-E", "-d", "-a"})
_XARGS_FLAG_OPTIONS = frozenset(
    {"-0", "-r", "-t", "-p", "--null", "--no-run-if-empty"}
)

# Programs that are dangerous when launched by ``xargs`` but are not denied
# on their own.
_XARGS_DENIED_PROGRAMS = frozenset(
    {
        "rm",
        "chmod",
        "chown",
        "shred",
        "unlink",
        "sh",
        "bash",
        "zsh",
        "dash",
        "env",
        "sudo",
        "xargs",
    }
)


def _is_in_place_token(token: str) -> bool:
    """Report whether an option token selects in-place editing (``-i``)."""
    if token == "--in-place" or token.startswith("--in-place="):
        return True
    if token.startswith("-i"):
        return True
    if not _is_short_flag_cluster(token):
        return False
    for char in token[1:]:
        if char in _PERL_VALUE_REST_OPTIONS or char in "eE" or char.isdigit():
            return False
        if char == "i":
            return True
    return False


def _perl_option_script(token: str) -> str | None:
    """Locate a ``perl`` script option inside a short-option token.

    Returns ``"attached"`` when the script follows ``-e``/``-E`` in the same
    token (``-e'...'``, ``-pie'...'``), ``"separate"`` when the script is the
    next argument (``-e``, ``-pie``), and ``None`` when the token carries no
    script option. Options that take the rest of the token as their value
    (``-Mstrict``, ``-Ilib``) end the scan.
    """
    if not token.startswith("-") or token.startswith("--") or len(token) < 2:
        return None
    body = token[1:]
    for pos, char in enumerate(body):
        if char in _PERL_VALUE_REST_OPTIONS:
            return None
        if char in "eE":
            return "attached" if pos + 1 < len(body) else "separate"
    return None


def _in_place_edit_outside_workspace(segment: str, workspace: Path) -> bool:
    """Report whether an in-place ``sed``/``perl`` edit targets outside the workspace.

    Only invocations that actually use in-place mode are checked. Script
    options (``-e``/``-f``/``--expression``/``--file`` for ``sed`` and
    ``-e``/``-E`` for ``perl``) consume their value; without one the first
    operand is the script. Every remaining file operand must stay inside the
    workspace and must not contain a shell expansion.
    """
    parsed = _program_and_args(segment)
    if parsed is None:
        return False
    name, args = parsed
    if name not in {"sed", "perl"}:
        return False

    in_place = False
    script_seen = False
    operands: list[str] = []
    options_done = False
    i = 0
    while i < len(args):
        token = args[i]
        if options_done:
            operands.append(token)
            i += 1
            continue
        if token == "--":
            options_done = True
            i += 1
            continue
        if name == "sed" and len(token) > 2 and token.startswith(("-e", "-f")):
            # The script is attached to the option (``-e's/a/b/'',
            # ``-fscript.sed``): it is a script option, not an in-place flag,
            # so letters inside the script name must not be scanned as flags.
            script_seen = True
            i += 1
            continue
        if _is_in_place_token(token):
            in_place = True
            i += 1
            continue
        if name == "sed" and token in {"-e", "-f", "--expression", "--file"}:
            script_seen = True
            i += 2
            continue
        if name == "sed" and (
            token.startswith("--expression=") or token.startswith("--file=")
        ):
            script_seen = True
            i += 1
            continue
        if name == "perl":
            script = _perl_option_script(token)
            if script is not None:
                script_seen = True
                i += 1 if script == "attached" else 2
                continue
        if token.startswith("-"):
            i += 1
            continue
        operands.append(token)
        i += 1

    if not in_place:
        return False
    if not script_seen and operands:
        operands = operands[1:]
    for operand in operands:
        target = operand.strip().strip("\"'")
        if not target:
            continue
        if "$" in target or "`" in target:
            return True
        if not _target_inside_workspace(target, workspace):
            return True
    return False


def _find_action_outside_workspace(segment: str, workspace: Path) -> bool:
    """Report whether ``find`` runs a destructive action with an outside root."""
    parsed = _program_and_args(segment)
    if parsed is None:
        return False
    name, args = parsed
    if name != "find":
        return False
    if not any(token in _FIND_ACTION_TOKENS for token in args):
        return False

    roots: list[str] = []
    index = 0
    # Leading global options may precede the roots (``find -L /etc ...``), so
    # they are skipped; ``-D`` consumes a separate value. ``--`` ends option
    # parsing.
    while index < len(args):
        token = args[index]
        if token == "--":
            index += 1
            break
        if token in _FIND_GLOBAL_FLAGS:
            index += 1
            continue
        if token == "-D":
            index += 2
            continue
        if token.startswith(("-D", "-O")):
            index += 1
            continue
        break
    while index < len(args):
        token = args[index]
        if token.startswith("-") or token in {"(", "!", ","}:
            break
        roots.append(token)
        index += 1
    if not roots:
        roots = ["."]

    for root in roots:
        target = root.strip().strip("\"'")
        if not target:
            continue
        if "$" in target or "`" in target:
            return True
        if not _target_inside_workspace(target, workspace):
            return True
    return False


def _xargs_with_write_program(segment: str, workspace: Path) -> bool:
    """Report whether ``xargs`` launches a denied or file-writing program.

    ``xargs`` options are skipped, then the first remaining token is the
    launched program. A write-path program, a denied program or an in-place
    ``sed``/``perl`` invocation is denied.
    """
    parsed = _program_and_args(segment)
    if parsed is None:
        return False
    name, args = parsed
    if name != "xargs":
        return False

    i = 0
    while i < len(args):
        token = args[i]
        if token == "--":
            i += 1
            break
        if token in _XARGS_VALUE_OPTIONS:
            i += 2
            continue
        if token in _XARGS_FLAG_OPTIONS:
            i += 1
            continue
        if token.startswith("-") and len(token) > 1:
            i += 1
            continue
        break

    if i >= len(args):
        return False
    program = os.path.basename(args[i]).lower()
    if (
        program in DENIED_PROGRAMS
        or program in _WRITE_PATH_PROGRAMS
        or program in _XARGS_DENIED_PROGRAMS
    ):
        return True
    return bool(
        program in {"sed", "perl"}
        and any(_is_in_place_token(token) for token in args[i + 1 :])
    )


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
    if _writes_outside_workspace(segment, workspace):
        return True

    # DENY in-place edits, destructive find actions and xargs write programs
    if _in_place_edit_outside_workspace(segment, workspace):
        return True
    if _find_action_outside_workspace(segment, workspace):
        return True
    return bool(_xargs_with_write_program(segment, workspace))


def _text_is_denied(text: str, workspace: Path, depth: int) -> bool:
    """Check a command - or the text of a command substitution - against the policy."""
    # POSIX shells drop backslash-newline continuations before tokenizing, so
    # join them first so both the whole-text and per-segment checks see the
    # resulting command.
    text = re.sub(r"\\\r?\n", "", text)
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
