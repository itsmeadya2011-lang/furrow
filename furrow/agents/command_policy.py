"""Command security policy for the worker agent.

This module contains the logic for validating shell commands against a security
policy. It checks for denied commands, dangerous programs, force pushes,
redirections outside the workspace, and command substitutions.

The policy is designed to be strict by default while allowing normal development
commands like running tests, linting, and file operations within the workspace.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Final

# Substring patterns that are denied in a command. They are matched
# case-insensitively against the normalized command.
DENIED_COMMANDS: Final[tuple[str, ...]] = (
    ":(){:|:&};:",
    "mkfs",
    "dd if=",
    "shutdown",
    "reboot",
    "> /dev/sda",
)

# Programs that are denied by name: the basename of the first token of a
# segment. ``mkfs`` variants (``mkfs.ext4`` and friends) are denied as well.
DENIED_PROGRAMS: Final[frozenset[str]] = frozenset(
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
TRUNCATED_STOP_REASONS: Final[frozenset[str]] = frozenset({"max_tokens", "length"})

# Maximum number of entries returned by the ``list_files`` tool.
MAX_LIST_ENTRIES: Final[int] = 500

# Maximum number of characters returned by the ``read_file`` tool.
MAX_READ_CHARS: Final[int] = 20000

# Shell separators a command is split on. A lone ``&`` backgrounds a command,
# so the command on each side is checked as its own segment.
_SEGMENT_SPLIT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:&&|\|\||;|&|(?<!>)\||\n|\r)+"
)

# Every spelling of the HOME variable. The look-ahead keeps "$HOME_DIR" and
# friends untouched.
_HOME_VAR: Final[str] = r"\$\{?home\}?(?![A-Za-z0-9_])"
_HOME_VAR_RE: Final[re.Pattern[str]] = re.compile(
    rf'"(?:{_HOME_VAR})"' rf"|'(?:{_HOME_VAR})'" rf"|{_HOME_VAR}",
    re.IGNORECASE,
)

# The target of a redirection: ">> file", ">| file" (clobber override) or
# "> file". ``&>`` and ``>&`` are caught because their ``>`` is matched here.
_REDIRECT_RE: Final[re.Pattern[str]] = re.compile(r"(?:>>|>\|?)\s*([^\s;|&<>]+)")

# How deep command substitutions are followed: a substitution may contain
# another substitution.
_MAX_SUBSTITUTION_DEPTH: Final[int] = 3

# Leading program names that are only wrappers: they run the command that
# follows them, so a dangerous ``rm``/``chmod`` hides behind them.
_WRAPPER_PROGRAMS: Final[frozenset[str]] = frozenset(
    {"sudo", "doas", "command", "env", "nohup", "exec"}
)

# Regex for a leading variable assignment such as ``FOO=1`` or ``PATH=/bin``.
_ASSIGNMENT_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Targets that discard output rather than write a real file.
_SPECIAL_WRITE_TARGETS: Final[frozenset[str]] = frozenset(
    {"/dev/null", "/dev/stdout", "/dev/stderr"}
)

# Programs that write to path arguments rather than via shell redirections.
_WRITE_PATH_PROGRAMS: Final[frozenset[str]] = frozenset({"tee"})


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
    cmd = re.sub(r"\s+", " ", cmd)
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
    VALUE_OPTS: dict[str, set[str]] = {
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


def _git_subcommand(args: list[str]) -> str | None:
    """Return the git subcommand, skipping global options that take a value."""
    VALUE_OPTS = {"-c", "-C", "--git-dir", "--work-tree", "--namespace", "--exec-path"}
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


def resolve_in_workspace(workspace: Path, path: str) -> Path | None:
    """Resolve a path relative to workspace, returning None if it escapes.

    The path is resolved relative to the workspace. If the resolved path
    is not within the workspace (or its parents), returns None.
    """
    try:
        workspace_resolved = workspace.resolve()
        path_obj = Path(path)
        resolved = path_obj if path_obj.is_absolute() else (workspace_resolved / path_obj)
        resolved = resolved.resolve()
        resolved.relative_to(workspace_resolved)
        return resolved
    except (ValueError, OSError, RuntimeError):
        return None


def is_command_denied(command: str, workspace: Path) -> bool:
    """Check if command is denied by security policy."""
    if not command or not command.strip():
        return False
    return _text_is_denied(command, workspace, depth=0)