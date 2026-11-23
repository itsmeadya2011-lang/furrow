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
_WRAPPER_PROGRAMS = frozenset(
    {
        "sudo",
        "doas",
        "command",
        "env",
        "nohup",
        "exec",
        "nice",
        "timeout",
        "stdbuf",
        "setsid",
        "chroot",
        "unshare",
    }
)

# Wrappers that consume one leading positional token before the real command:
# ``timeout`` takes a DURATION and ``chroot`` a NEWROOT. The count is applied
# after the wrapper's own options have been skipped.
_WRAPPER_SKIP_POSITIONALS = {"timeout": 1, "chroot": 1}

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
    Wrapper option flags are skipped too. Wrappers that consume a leading
    positional token (``timeout``'s DURATION, ``chroot``'s NEWROOT) skip it as
    well, so the reported program is the one that actually executes.
    """
    tokens = _shell_tokens(segment.strip())
    if not tokens:
        return None
    VALUE_OPTS = {
        "sudo": {"-u", "--user", "-g", "--group", "-p", "--prompt"},
        "doas": {"-u", "--user", "-g", "--group", "-p", "--prompt"},
        "env": {"-u", "--unset", "-C", "--chdir"},
        "nice": {"-n", "--adjustment"},
        "stdbuf": {"-o", "-e", "-i", "--output", "--error", "--input"},
        "timeout": {"-s", "--signal", "-k", "--kill-after"},
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
            # Skip a leading positional the wrapper itself consumes (the
            # DURATION of ``timeout`` or the NEWROOT of ``chroot``).
            skip = _WRAPPER_SKIP_POSITIONALS.get(base, 0)
            for _ in range(skip):
                if i >= len(tokens):
                    break
                i += 1
            continue
        break
    if i >= len(tokens):
        return None
    return os.path.basename(tokens[i]).lower(), tokens[i + 1 :]


# Shells that execute the string argument that follows ``-c``/``--command``.
_SHELL_PROGRAMS = frozenset({"bash", "sh", "dash", "zsh", "ksh", "ash"})


def _shell_command_payloads(segment: str) -> list[str]:
    """Return the command text a shell ``-c`` or ``eval`` will execute.

    ``bash``/``sh``/... run the argument that follows ``-c``/``--command``,
    including combined or attached spellings such as ``-lc`` or ``-c'cmd'``.
    ``eval`` concatenates all of its arguments with spaces and runs the
    result. The effective program is resolved through wrapper programs first,
    so ``sudo bash -c ...`` is inspected too.
    """
    parsed = _program_and_args(segment)
    if parsed is None:
        return []
    name, args = parsed
    if name in _SHELL_PROGRAMS:
        payloads: list[str] = []
        i = 0
        while i < len(args):
            token = args[i]
            if token in ("-c", "--command"):
                if i + 1 < len(args):
                    payloads.append(args[i + 1])
                break
            if token.startswith("--command="):
                payloads.append(token.split("=", 1)[1])
                break
            if _is_short_flag_cluster(token) and "c" in token[1:]:
                # Combined/attached ``-c``: ``-lc``/``-ec`` take the next
                # token, ``-c'cmd'`` carries the command in this token.
                remainder = token[1:].split("c", 1)[1]
                if remainder:
                    payloads.append(remainder)
                elif i + 1 < len(args):
                    payloads.append(args[i + 1])
                break
            i += 1
        return payloads
    if name == "eval":
        joined = " ".join(token for token in args if token != "--")
        return [joined] if joined.strip() else []
    return []


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

    Every target is denied if it contains a shell expansion (``$`` or a
    backtick) or resolves outside the workspace, whether or not ``-r``/``-f``
    is given. The option terminator (``--``) and option clusters are still
    skipped when collecting targets.
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
        if token.startswith("-"):
            continue
        targets.append(token)
    return any(_denies_target(target, workspace) for target in targets)


def _is_dangerous_chmod(args: list[str], workspace: Path) -> bool:
    """Report whether a ``chmod`` invocation targets outside the workspace.

    Every operand is checked, with or without ``-r``/``-R``. The leading mode
    token (e.g. ``777``) resolves as a harmless relative path, so it never
    triggers a denial.
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
        if token.startswith("-"):
            continue
        targets.append(token)
    return any(_denies_target(target, workspace) for target in targets)


# Filesystem-mutating programs whose path operands are workspace-scoped.
_FS_MUTATION_PROGRAMS = frozenset({"chown", "touch", "mkdir"})

# Value-taking options of the filesystem-mutating programs. Their value is not
# a path operand.
_FS_MUTATION_VALUE_OPTS = {
    "chown": {"--reference"},
    "touch": {"-r", "--reference", "-d", "--date", "-t"},
    "mkdir": {"-m", "--mode"},
}


def _is_dangerous_fs_mutation(segment: str, workspace: Path) -> bool:
    """Report whether ``chown``/``touch``/``mkdir`` target outside the workspace.

    Wrappers are resolved first, so ``sudo chown root /etc/passwd`` is caught.
    ``ln`` is handled separately as a destination-aware write.
    """
    parsed = _program_and_args(segment)
    if parsed is None:
        return False
    name, args = parsed
    if name not in _FS_MUTATION_PROGRAMS:
        return False
    operands = _positional_operands(name, args, _FS_MUTATION_VALUE_OPTS.get(name))
    return any(_denies_target(target, workspace) for target in operands)


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

# Programs that modify files in place, so their path operands are write
# targets. ``sed`` only writes when ``-i``/``--in-place`` is present.
_INPLACE_WRITE_PROGRAMS = frozenset({"shred", "truncate", "sed"})

# Value-taking options of the in-place writers. Their value is not a path.
_INPLACE_VALUE_OPTS = {
    "shred": {"-n", "--iterations", "-s", "--size", "--random-source"},
    "truncate": {"-s", "--size"},
    "sed": {"-e", "--expression", "-f", "--file"},
}

# curl/wget options whose value is a write path.
_CURL_VALUE_OPTS = {"-o", "--output", "--output-dir"}
_WGET_VALUE_OPTS = {"-O", "--output-document"}

# cp/mv/install/ln options whose value is not a positional operand.
_DEST_VALUE_OPTS = {
    "cp": {"-t", "--target-directory", "-S", "--suffix", "--context"},
    "mv": {"-t", "--target-directory", "-S", "--suffix", "--context"},
    "install": {
        "-t",
        "--target-directory",
        "-m",
        "--mode",
        "-o",
        "--owner",
        "-g",
        "--group",
        "-S",
        "--suffix",
        "--context",
    },
    "ln": {"-t", "--target-directory", "-S", "--suffix"},
}
_DEST_PROGRAMS = frozenset({"cp", "mv", "install", "ln"})


def _positional_operands(
    name: str, args: list[str], value_opts: set[str] | None = None
) -> list[str]:
    """Return the positional (non-option) operands of a program's arguments.

    Options and their values are skipped, so ``-s 0`` contributes no operand
    and ``--size=0`` is consumed as a single option token. ``value_opts``
    defaults to the in-place writer options for ``name``.
    """
    if value_opts is None:
        value_opts = _INPLACE_VALUE_OPTS.get(name, set())
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
        if token.startswith("--"):
            if "=" in token or token not in value_opts:
                i += 1
            else:
                i += 2
            continue
        if token.startswith("-") and token != "-":
            i += 2 if token in value_opts else 1
            continue
        operands.append(token)
        i += 1
    return operands


def _sed_in_place(args: list[str]) -> bool:
    """Report whether a ``sed`` invocation edits files in place (``-i``)."""
    for token in args:
        if token == "--":
            break
        if token == "--in-place" or token.startswith("--in-place="):
            return True
        if _is_short_flag_cluster(token) and "i" in token[1:]:
            return True
    return False


def _sed_script_via_option(args: list[str]) -> bool:
    """Report whether a ``sed`` script was given with ``-e``/``-f``.

    When it was, the first positional operand is a file, not the script.
    """
    for token in args:
        if token == "--":
            break
        if token.startswith("--"):
            if token in ("--expression", "--file") or token.startswith(
                ("--expression=", "--file=")
            ):
                return True
            continue
        if token.startswith("-") and token != "-" and ("e" in token[1:] or "f" in token[1:]):
            return True
    return False


def _option_values(value_opts: set[str], args: list[str]) -> list[str]:
    """Return the values of options that take one, in both ``-o V`` and ``--o=V``."""
    values: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--":
            break
        if token.startswith("--"):
            if "=" in token:
                if token.split("=", 1)[0] in value_opts:
                    values.append(token.split("=", 1)[1])
                i += 1
                continue
            if token in value_opts:
                if i + 1 < len(args):
                    values.append(args[i + 1])
                    i += 2
                else:
                    i += 1
                continue
            i += 1
            continue
        if token.startswith("-") and token != "-":
            if token in value_opts:
                if i + 1 < len(args):
                    values.append(args[i + 1])
                    i += 2
                else:
                    i += 1
                continue
            # attached short-option value, e.g. ``-o/tmp/x`` or ``-O/tmp/x``
            attached = [
                opt
                for opt in value_opts
                if not opt.startswith("--") and len(token) > len(opt) and token.startswith(opt)
            ]
            if attached:
                opt = max(attached, key=len)
                values.append(token[len(opt) :])
                i += 1
                continue
            i += 1
            continue
        i += 1
    return values


def _destination_target(name: str, args: list[str]) -> str | None:
    """Return the destination path of a ``cp``/``mv``/``install`` invocation.

    The value of ``-t``/``--target-directory`` wins when present; otherwise the
    last positional operand is the destination. Option values are skipped so
    that e.g. ``install -m 644 src dst`` resolves to ``dst``.
    """
    value_opts = _DEST_VALUE_OPTS.get(name, set())
    positionals: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--":
            positionals.extend(args[i + 1 :])
            break
        if token.startswith("--"):
            key = token.split("=", 1)[0]
            if "=" in token:
                if key == "--target-directory":
                    return token.split("=", 1)[1]
                i += 1
                continue
            if token == "--target-directory":
                return args[i + 1] if i + 1 < len(args) else None
            i += 2 if token in value_opts else 1
            continue
        if token.startswith("-") and token != "-":
            if token == "-t":
                return args[i + 1] if i + 1 < len(args) else None
            if token.startswith("-t") and len(token) > 2:
                return token[2:]
            i += 2 if token in value_opts else 1
            continue
        positionals.append(token)
        i += 1
    if not positionals:
        return None
    return positionals[-1]


def _denies_target(target: str, workspace: Path) -> bool:
    """Report whether a write target is denied by the workspace policy.

    Empty targets, ``/dev/null`` and friends are allowed; a target that still
    contains a shell expansion (``$`` or a backtick) or resolves outside the
    workspace is denied.
    """
    cleaned = target.strip().strip("\"'")
    if not cleaned or cleaned in _SPECIAL_WRITE_TARGETS:
        return False
    if "$" in cleaned or "`" in cleaned:
        return True
    return not _target_inside_workspace(cleaned, workspace)


def _write_path_targets(name: str, args: list[str]) -> list[str] | None:
    """Return the path operands a program writes, or ``None`` if it does not.

    ``tee`` writes every operand; ``shred``/``truncate`` write every operand;
    ``sed`` writes its file operands only with ``-i`` (skipping the script);
    ``curl``/``wget`` write the value of their output option; and
    ``cp``/``mv``/``install`` write their destination.
    """
    if name in _INPLACE_WRITE_PROGRAMS:
        if name == "sed":
            if not _sed_in_place(args):
                return None
            operands = _positional_operands(name, args)
            return operands if _sed_script_via_option(args) else operands[1:]
        return _positional_operands(name, args)
    if name in _WRITE_PATH_PROGRAMS:
        return _positional_operands(name, args)
    if name == "curl":
        return _option_values(_CURL_VALUE_OPTS, args)
    if name == "wget":
        return _option_values(_WGET_VALUE_OPTS, args)
    if name in _DEST_PROGRAMS:
        dest = _destination_target(name, args)
        return [dest] if dest else []
    return None


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
    redirections (``tee``), edit files in place (``shred``/``truncate``/
    ``sed -i``), write through an output option (``curl``/``wget``), or copy
    to a destination (``cp``/``mv``/``install``). A path outside the workspace
    or one that still contains a shell expansion (``$`` or a backtick) is
    denied. A program with no write path is allowed.
    """
    parsed = _program_and_args(segment)
    if parsed is None:
        return False
    name, args = parsed
    targets = _write_path_targets(name, args)
    if targets is None:
        return False
    return any(_denies_target(target, workspace) for target in targets)


def _is_dangerous_git_clean(segment: str) -> bool:
    """Report whether a segment is a ``git clean`` that destroys work.

    A force flag (``-f`` in a short cluster, ``--force`` or ``-ff``) without a
    dry run (``-n``/``--dry-run``) removes untracked and ignored files.
    """
    parsed = _program_and_args(segment)
    if parsed is None:
        return False
    name, args = parsed
    if name != "git":
        return False
    if _git_subcommand(args) != "clean":
        return False
    force = False
    dry_run = False
    for token in args:
        if token == "--":
            break
        lowered = token.lower()
        if lowered == "--force":
            force = True
            continue
        if lowered in ("-n", "--dry-run"):
            dry_run = True
            continue
        if _is_short_flag_cluster(token):
            if _short_cluster_has(token, "f"):
                force = True
            if _short_cluster_has(token, "n"):
                dry_run = True
    return force and not dry_run


def _find_start_paths(args: list[str]) -> list[str]:
    """Return the start paths of a ``find`` invocation (default ``.``).

    Leading options (``-H``/``-L``/``-P`` flags, and ``-D``/``-O`` with a
    separate or attached value) are skipped before the start paths, which are
    the non-option tokens up to the first expression token.
    """
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--":
            i += 1
            break
        if token in ("-H", "-L", "-P"):
            i += 1
            continue
        if token in ("-D", "-O"):
            i += 2
            continue
        if token.startswith(("-D", "-O")):
            # attached option value, e.g. ``-O3`` or ``-Dexec``
            i += 1
            continue
        break
    paths: list[str] = []
    while i < len(args) and not args[i].startswith("-"):
        paths.append(args[i])
        i += 1
    return paths or ["."]


def _is_dangerous_find(segment: str, workspace: Path) -> bool:
    """Report whether a ``find`` invocation runs commands or deletes broadly.

    ``-exec``/``-execdir``/``-ok``/``-okdir`` run an arbitrary command and are
    always denied. ``-delete`` is denied when a start path resolves outside the
    workspace (or still contains a shell expansion).
    """
    parsed = _program_and_args(segment)
    if parsed is None:
        return False
    name, args = parsed
    if name != "find":
        return False
    if any(token in ("-exec", "-execdir", "-ok", "-okdir") for token in args):
        return True
    if "-delete" in args:
        for path in _find_start_paths(args):
            if _denies_target(path, workspace):
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

    # DENY a force push or a destructive ``git clean``
    if _is_force_push(segment, normalized) or _is_dangerous_git_clean(segment):
        return True

    # DENY ``find`` that runs commands or deletes outside the workspace
    if _is_dangerous_find(segment, workspace):
        return True

    # DENY ``chown``/``touch``/``mkdir`` that target outside the workspace
    if _is_dangerous_fs_mutation(segment, workspace):
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
    segments = _split_command_segments(text)
    if any(_is_dangerous_segment(segment, workspace) for segment in segments):
        return True
    # The shell runs substituted text and shell ``-c``/``eval`` payloads, so
    # they are checked with the same rules, up to the recursion limit.
    if depth < _MAX_SUBSTITUTION_DEPTH:
        for substitution in _extract_command_substitutions(text):
            if _text_is_denied(substitution, workspace, depth + 1):
                return True
        for segment in segments:
            for payload in _shell_command_payloads(segment):
                if _text_is_denied(payload, workspace, depth + 1):
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
