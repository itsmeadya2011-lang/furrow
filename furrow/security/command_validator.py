"""Command validation and security utilities for Furrow.

This module provides a unified, configurable security layer for validating
shell commands before execution. It replaces the inline validation logic
in the worker agent with a more maintainable and testable approach.
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

# Shell separators a command is split on.
_SEGMENT_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"(?:&&|\|\||;|\||\n|\r)+")

# Every spelling of the HOME variable. The look-ahead keeps "$HOME_DIR" and
# friends untouched.
_HOME_VAR = r"\$\{?home\}?(?![A-Za-z0-9_])"
_HOME_VAR_RE: Final[re.Pattern[str]] = re.compile(
    rf'"(?:{_HOME_VAR})"' rf"|'(?:{_HOME_VAR})'" rf"|{_HOME_VAR}",
    re.IGNORECASE,
)

# The target of a redirection: "> file" or ">> file".
_REDIRECT_RE: Final[re.Pattern[str]] = re.compile(r">>?\s*([^\s;|&<>]+)")

# How deep command substitutions are followed: a substitution may contain
# another substitution.
_MAX_SUBSTITUTION_DEPTH: Final[int] = 3

# Dangerous path patterns that should never be accessed
_DANGEROUS_PATHS: Final[tuple[str, ...]] = (
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    "/proc/self/environ",
    "/proc/version",
    "/sys/firmware",
    "/dev/mem",
    "/dev/kmem",
    "/dev/port",
    "/.dockerenv",
    "/.dockerinit",
    "/run/secrets",
    "/var/run/secrets",
    "/secrets",
)

# Patterns that indicate sensitive file access
_SENSITIVE_PATH_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"/\.ssh/"),
    re.compile(r"/\.aws/"),
    re.compile(r"/\.gnupg/"),
    re.compile(r"/\.docker/"),
    re.compile(r"/\.kube/"),
    re.compile(r"/\.config/gcloud/"),
    re.compile(r"/\.azure/"),
    re.compile(r"id_rsa"),
    re.compile(r"id_ed25519"),
    re.compile(r"\.pem$"),
    re.compile(r"\.key$"),
    re.compile(r"\.crt$"),
    re.compile(r"secret"),
    re.compile(r"password"),
    re.compile(r"token"),
    re.compile(r"api[_-]?key"),
)


class PathValidator:
    """Validates file paths against security policies."""

    def __init__(
        self,
        workspace: Path,
        allowed_paths: list[Path] | None = None,
        blocked_paths: list[Path] | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.allowed_paths = [p.resolve() for p in (allowed_paths or [])]
        self.blocked_paths = [p.resolve() for p in (blocked_paths or [])]
        # Add default blocked paths
        self.blocked_paths.extend([Path(p).resolve() for p in _DANGEROUS_PATHS])

    def is_path_allowed(self, path: Path) -> tuple[bool, str | None]:
        """Check if a path is allowed to be accessed.

        Returns:
            Tuple of (allowed, reason). If allowed is False, reason explains why.
        """
        try:
            resolved = path.resolve()
        except Exception as e:
            return False, f"Cannot resolve path: {e}"

        # Check if path is within workspace
        try:
            resolved.relative_to(self.workspace)
        except ValueError:
            # Path is outside workspace - check if explicitly allowed
            for allowed in self.allowed_paths:
                try:
                    resolved.relative_to(allowed)
                    return True, None
                except ValueError:
                    continue
            return False, "Path escapes workspace"

        # Check blocked paths
        for blocked in self.blocked_paths:
            try:
                resolved.relative_to(blocked)
                return False, f"Path is in blocked directory: {blocked}"
            except ValueError:
                continue

        # Check sensitive path patterns
        path_str = str(resolved)
        for pattern in _SENSITIVE_PATH_PATTERNS:
            if pattern.search(path_str):
                return False, f"Path matches sensitive pattern: {pattern.pattern}"

        return True, None

    def validate_read(self, path: Path) -> tuple[bool, str | None]:
        """Validate a path for reading."""
        return self.is_path_allowed(path)

    def validate_write(self, path: Path) -> tuple[bool, str | None]:
        """Validate a path for writing."""
        allowed, reason = self.is_path_allowed(path)
        if not allowed:
            return False, reason
        # Additional checks for write operations
        if path.is_dir():
            return False, "Cannot write to a directory"
        return True, None

    def validate_list(self, path: Path) -> tuple[bool, str | None]:
        """Validate a path for listing."""
        return self.is_path_allowed(path)


class CommandValidator:
    """Validates shell commands against a security policy.

    The validator normalizes commands, checks for denied patterns and programs,
    detects force pushes, and ensures redirections stay within the workspace.
    """

    def __init__(
        self,
        workspace: Path,
        denied_commands: tuple[str, ...] | None = None,
        denied_programs: frozenset[str] | None = None,
        max_substitution_depth: int = _MAX_SUBSTITUTION_DEPTH,
        path_validator: PathValidator | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.denied_commands = denied_commands if denied_commands is not None else DENIED_COMMANDS
        self.denied_programs = denied_programs if denied_programs is not None else DENIED_PROGRAMS
        self.max_substitution_depth = max_substitution_depth
        self.path_validator = path_validator or PathValidator(workspace)

    @staticmethod
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

    @staticmethod
    def _split_command_segments(command: str) -> list[str]:
        """Split a command into segments on the shell separators ``;``, ``&&``,
        ``||``, ``|`` and newlines."""
        if not command:
            return []
        segments = _SEGMENT_SPLIT_RE.split(command)
        return [s for s in segments if s.strip()]  # Remove empty segments

    @staticmethod
    def _shell_tokens(segment: str) -> list[str]:
        """Split a segment into shell tokens.

        A segment that cannot be parsed (unbalanced quotes, for example) falls
        back to a plain whitespace split.
        """
        try:
            return shlex.split(segment)
        except ValueError:
            return segment.split()

    @staticmethod
    def _get_program_basename(segment: str) -> str | None:
        """Get basename of first token in segment."""
        tokens = CommandValidator._shell_tokens(segment.strip())
        if not tokens:
            return None
        return os.path.basename(tokens[0])

    @staticmethod
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

    @staticmethod
    def _expand_home(target: str) -> str:
        """Expand ``~`` and the HOME variable of a redirection target."""
        home = os.path.expanduser("~")
        return os.path.expanduser(_HOME_VAR_RE.sub(home, target))

    def _target_inside_workspace(self, target: str) -> bool:
        """Resolve a redirection target and report whether it stays in the workspace.

        An absolute target is resolved as it is, a relative one against the
        workspace.
        """
        base = self.workspace
        path = Path(self._expand_home(target))
        resolved = path if path.is_absolute() else base / path
        try:
            resolved.resolve().relative_to(base)
        except ValueError:
            return False
        return True

    def _redirects_outside_workspace(self, segment: str) -> bool:
        """Report whether a segment redirects output outside of the workspace."""
        for match in _REDIRECT_RE.finditer(segment):
            target = match.group(1).strip().strip("\"'")
            if not target:
                continue
            if not self._target_inside_workspace(target):
                return True
        return False

    @staticmethod
    def _is_force_push(segment: str, normalized: str) -> bool:
        """Report whether a segment is a git force push."""
        if "git" not in normalized or "push" not in normalized:
            return False
        # Covers --force, --force-with-lease and --force-with-lease=<ref>.
        if "--force" in normalized or "--force-with-lease" in normalized:
            return True
        # A short -f flag only counts as a shell token, so that ordinary
        # commands (e.g. "ls -f") are not mistaken for a force push.
        return any(token.lower() == "-f" for token in CommandValidator._shell_tokens(segment))

    def _is_dangerous_segment(self, segment: str) -> bool:
        """Report whether a single command segment violates the command policy."""
        normalized = self._normalize_command(segment)
        if not normalized:
            return False

        # DENY by PROGRAM NAME
        program = self._get_program_basename(segment)
        if program:
            program = program.lower()
            if program in self.denied_programs or program.startswith("mkfs."):
                return True

        # DENY by PATTERN
        if any(denied.lower() in normalized for denied in self.denied_commands):
            return True

        # DENY a force push
        if self._is_force_push(segment, normalized):
            return True

        # DENY a redirection that escapes the workspace
        return self._redirects_outside_workspace(segment)

    def _text_is_denied(self, text: str, depth: int) -> bool:
        """Check a command - or the text of a command substitution - against the policy."""
        normalized = self._normalize_command(text)
        # The whole text is checked against the pattern list as well, so that a
        # denied pattern cannot hide in an unusual position (inside quotes, in
        # the middle of a pipeline, ...).
        if any(denied.lower() in normalized for denied in self.denied_commands):
            return True
        # Every segment of the text is checked on its own.
        if any(self._is_dangerous_segment(segment) for segment in self._split_command_segments(text)):
            return True
        # The shell runs the substituted text, so it is checked as well.
        if depth < self.max_substitution_depth:
            for substitution in self._extract_command_substitutions(text):
                if self._text_is_denied(substitution, depth + 1):
                    return True
        return False

    def is_command_denied(self, command: str) -> bool:
        """Check if command is denied by security policy.

        Args:
            command: The shell command to validate.

        Returns:
            True if the command is denied, False otherwise.
        """
        if not command or not command.strip():
            return False
        return self._text_is_denied(command, depth=0)

    def validate_command_with_reason(self, command: str) -> tuple[bool, str | None]:
        """Check if command is denied and return the reason.

        Returns:
            Tuple of (denied, reason). If denied is True, reason explains why.
        """
        if not command or not command.strip():
            return False, None

        normalized = self._normalize_command(command)

        # Check denied programs
        for segment in self._split_command_segments(command):
            program = self._get_program_basename(segment)
            if program:
                program = program.lower()
                if program in self.denied_programs or program.startswith("mkfs."):
                    return True, f"Denied program: {program}"

        # Check denied patterns
        for denied in self.denied_commands:
            if denied.lower() in normalized:
                return True, f"Denied pattern: {denied}"

        # Check force push
        for segment in self._split_command_segments(command):
            if self._is_force_push(segment, normalized):
                return True, "Git force push detected"

        # Check redirections
        for segment in self._split_command_segments(command):
            if self._redirects_outside_workspace(segment):
                return True, "Redirection outside workspace"

        # Check command substitutions
        if self._text_is_denied(command, depth=0):
            return True, "Dangerous command substitution"

        return False, None


def resolve_in_workspace(workspace: Path, path: str) -> Path | None:
    """Return the absolute path for `path` inside `workspace`, or None when it escapes."""
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