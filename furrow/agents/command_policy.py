"""Command security policy for the Furrow worker agent.

This module contains the hardened command execution policy that prevents
destructive operations while allowing normal development commands.

The policy works by:
1. Normalizing commands (lowercase, collapse whitespace, canonicalize HOME variable)
2. Splitting commands into segments on shell separators (;, &&, ||, |, newlines)
3. Checking each segment against denied patterns and programs
4. Recursively checking command substitutions ($(...), `...`)
5. Blocking force pushes and redirections that escape the workspace

Public API:
    is_command_denied: Check if a command violates the security policy
    resolve_in_workspace: Resolve a path within the workspace, returning None if it escapes
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path


# Substring patterns that are denied in a command. They are matched
# case-insensitively against the normalized command.
DENIED_COMMANDS = (
    "rm -rf /",
    "rm -fr /",
    "rm -rf /*",
    "rm -rf ~",
    "rm -rf $home",
    ":(){:|:&};:",
    "mkfs",
    "dd if=",
    "shutdown",
    "reboot",
    "chmod -R 777 /",
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

# Shell separators a command is split on.
_SEGMENT_SPLIT_RE = re.compile(r"(?:&&|\|\||;|\||\n|\r)+")

# Every spelling of the HOME variable. The look-ahead keeps "$HOME_DIR" and
# friends untouched.
_HOME_VAR = r"\$\{?home\}?(?![A-Za-z0-9_])"
_HOME_VAR_RE = re.compile(
    rf'"(?:{_HOME_VAR})"' rf"|'(?:{_HOME_VAR})'" rf"|{_HOME_VAR}",
    re.IGNORECASE,
)

# The target of a redirection: "> file" or ">> file".
_REDIRECT_RE = re.compile(r">>?\s*([^\s;|&<>]+)")

# How deep command substitutions are followed: a substitution may contain
# another substitution.
_MAX_SUBSTITUTION_DEPTH = 3


def _normalize_command(command: str) -> str:
    """Normalize a command so that it can be matched.

    The command is lower-cased, runs of whitespace are collapsed into a
    single space, the quotes around the whole command are removed and every
    spelling of the HOME variable (``$HOME``, ``${HOME}``, ``"$HOME"``,
    ``'$HOME'``, ``$home``, ``${home}``) is canonicalised to the single
    token ``$home``.

    Args:
        command: The command to normalize.

    Returns:
        The normalized command string.
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
    ``||``, ``|`` and newlines.

    Args:
        command: The command to split.

    Returns:
        A list of non-empty command segments.
    """
    if not command:
        return []
    segments = _SEGMENT_SPLIT_RE.split(command)
    return [s for s in segments if s.strip()]  # Remove empty segments


def _shell_tokens(segment: str) -> list[str]:
    """Split a segment into shell tokens.

    A segment that cannot be parsed (unbalanced quotes, for example) falls
    back to a plain whitespace split.

    Args:
        segment: The command segment to tokenize.

    Returns:
        A list of shell tokens.
    """
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _get_program_basename(segment: str) -> str | None:
    """Get basename of first token in segment.

    Args:
        segment: The command segment.

    Returns:
        The basename of the first token, or None if the segment is empty.
    """
    tokens = _shell_tokens(segment.strip())
    if not tokens:
        return None
    return os.path.basename(tokens[0])


def _extract_command_substitutions(command: str) -> list[str]:
    """Return the text of every ``$(...)`` and `` `...` `` substitution.

    The shell runs the substituted text, so it has to be checked with the
    very same rules as the command itself.

    Args:
        command: The command to extract substitutions from.

    Returns:
        A list of substitution texts.
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
    """Expand ``~`` and the HOME variable of a redirection target.

    Args:
        target: The redirection target.

    Returns:
        The expanded target.
    """
    home = os.path.expanduser("~")
    return os.path.expanduser(_HOME_VAR_RE.sub(home, target))


def _target_inside_workspace(target: str, workspace: Path) -> bool:
    """Resolve a redirection target and report whether it stays in the workspace.

    An absolute target is resolved as it is, a relative one against the
    workspace.

    Args:
        target: The redirection target.
        workspace: The workspace directory.

    Returns:
        True if the target resolves inside the workspace, False otherwise.
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
    """Report whether a segment redirects output outside of the workspace.

    Args:
        segment: The command segment to check.
        workspace: The workspace directory.

    Returns:
        True if the segment contains a redirection that escapes the workspace.
    """
    for match in _REDIRECT_RE.finditer(segment):
        target = match.group(1).strip().strip("\"'")
        if not target:
            continue
        if not _target_inside_workspace(target, workspace):
            return True
    return False


def _is_force_push(segment: str, normalized: str) -> bool:
    """Report whether a segment is a git force push.

    Args:
        segment: The original command segment.
        normalized: The normalized command segment.

    Returns:
        True if the segment is a git force push.
    """
    if "git" not in normalized or "push" not in normalized:
        return False
    # Covers --force, --force-with-lease and --force-with-lease=<ref>.
    if "--force" in normalized or "--force-with-lease" in normalized:
        return True
    # A short -f flag only counts as a shell token, so that ordinary
    # commands (e.g. "ls -f") are not mistaken for a force push.
    return any(token.lower() == "-f" for token in _shell_tokens(segment))


def _is_dangerous_segment(segment: str, workspace: Path) -> bool:
    """Report whether a single command segment violates the command policy.

    Args:
        segment: The command segment to check.
        workspace: The workspace directory.

    Returns:
        True if the segment is dangerous.
    """
    normalized = _normalize_command(segment)
    if not normalized:
        return False

    # DENY by PROGRAM NAME
    program = _get_program_basename(segment)
    if program:
        program = program.lower()
        if program in DENIED_PROGRAMS or program.startswith("mkfs."):
            return True

    # DENY by PATTERN
    if any(denied.lower() in normalized for denied in DENIED_COMMANDS):
        return True

    # DENY a force push
    if _is_force_push(segment, normalized):
        return True

    # DENY a redirection that escapes the workspace
    return bool(_redirects_outside_workspace(segment, workspace))


def _text_is_denied(text: str, workspace: Path, depth: int) -> bool:
    """Check a command - or the text of a command substitution - against the policy.

    Args:
        text: The command or substitution text to check.
        workspace: The workspace directory.
        depth: The current recursion depth for command substitutions.

    Returns:
        True if the text is denied.
    """
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
    """Check if command is denied by security policy.

    Args:
        command: The shell command to check.
        workspace: The workspace directory that commands must not escape.

    Returns:
        True if the command is denied, False otherwise.
    """
    if not command or not command.strip():
        return False
    return _text_is_denied(command, workspace, depth=0)


def resolve_in_workspace(workspace: Path, path: str) -> Path | None:
    """Return the absolute path for `path` inside `workspace`, or None when it escapes.

    Args:
        workspace: The workspace directory.
        path: The path to resolve (relative or absolute).

    Returns:
        The resolved absolute path if it stays within the workspace, None otherwise.
    """
    workspace = workspace.resolve()
    if not path:
        return None
    p = Path(path)
    if p.is_absolute():
        try:
            p.relative_to(workspace)
        except ValueError:
            return None
        return p
    # reject traversal
    candidate = (workspace / p).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError:
        return None
    return candidate