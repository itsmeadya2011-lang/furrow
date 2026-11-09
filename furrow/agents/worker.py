from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shlex
import signal
import time
from pathlib import Path
from typing import Any

from furrow.agents.prompts import WORKER_SYSTEM, WORKER_TOOLS
from furrow.config import Settings, TaskModel, WorkerResult
from furrow.config import settings as global_settings
from furrow.llm import LLMClient, LLMResponse, ToolCall
from furrow.logging import get_logger, set_task_id

# Substring patterns that are denied in a command. They are matched
# case-insensitively against the normalized command.
DENIED_COMMANDS = (
    # Mass deletion
    "rm -rf /",
    "rm -fr /",
    "rm -rf /*",
    "rm -rf ~",
    "rm -rf $home",
    "rm -rf *",
    "rm -fr *",
    # Fork bombs
    ":(){:|:&};:",
    # Disk operations
    "mkfs",
    "dd if=",
    "fdisk",
    "parted",
    "mkfs.",
    # System control
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init 0",
    "init 6",
    # Permission escalation
    "chmod -R 777 /",
    "chmod 777 /",
    "chown -R root /",
    # Direct device writes
    "> /dev/sda",
    "> /dev/nvme",
    "> /dev/mmcblk",
    # Network attacks
    "curl | sh",
    "wget | sh",
    "curl | bash",
    "wget | bash",
    "nc -l",
    "netcat -l",
    # Process manipulation
    "kill -9 -1",
    "pkill -9",
    # Kernel modules
    "insmod",
    "rmmod",
    "modprobe",
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
        "mkfs.ext4",
        "mkfs.xfs",
        "mkfs.btrfs",
        "mkfs.fat",
        "mkfs.vfat",
        "mkfs.ntfs",
        "mkswap",
        "swapon",
        "swapoff",
        "mount",
        "umount",
        "losetup",
        "cryptsetup",
        "lvm",
        "vgcreate",
        "lvcreate",
        "pvcreate",
        "mdadm",
        "fdisk",
        "cfdisk",
        "sfdisk",
        "partprobe",
        "kpartx",
        "dmsetup",
        "insmod",
        "rmmod",
        "modprobe",
        "depmod",
        "lsmod",
        "kexec",
        "kdump",
        "systemctl",
        "service",
        "update-rc.d",
        "chkconfig",
        "iptables",
        "ip6tables",
        "nft",
        "ufw",
        "firewall-cmd",
        "sshd",
        "sshd-keygen",
        "ssh-keygen",
        "ssh-copy-id",
        "ssh-agent",
        "ssh-add",
        "gpg",
        "gpg2",
        "openssl",
        "keytool",
        "certbot",
        "acme.sh",
        "letsencrypt",
        "passwd",
        "usermod",
        "useradd",
        "userdel",
        "groupmod",
        "groupadd",
        "groupdel",
        "chpasswd",
        "newusers",
        "vipw",
        "vigr",
        "visudo",
        "sudo",
        "su",
        "runuser",
        "pkexec",
        "doas",
        "setfacl",
        "getfacl",
        "chattr",
        "lsattr",
        "capsh",
        "setcap",
        "getcap",
        "apparmor_parser",
        "aa-enforce",
        "aa-complain",
        "aa-disable",
        "selinux",
        "setenforce",
        "getenforce",
        "auditctl",
        "ausearch",
        "aureport",
        "systemd-analyze",
        "journalctl",
        "loginctl",
        "machinectl",
        "coredumpctl",
        "hostnamectl",
        "localectl",
        "timedatectl",
        "kernelstub",
        "grub-install",
        "grub-mkconfig",
        "update-grub",
        "efibootmgr",
        "fwupdmgr",
        "fwupdate",
        "dmidecode",
        "lscpu",
        "lsblk",
        "lspci",
        "lsusb",
        "lshw",
        "inxi",
        "hwinfo",
        "smartctl",
        "hdparm",
        "nvme",
        "nvme-cli",
        "lsblk",
        "blkid",
        "lsblk",
        "findmnt",
        "mountpoint",
        "lsblk",
    }
)

# Maximum number of entries returned by the ``list_files`` tool.
MAX_LIST_ENTRIES = 500

# Programs that are explicitly allowed even if they match a denied pattern.
# These are common development tools that must be usable.
ALLOWED_PROGRAMS = frozenset(
    {
        "git",
        "npm",
        "npx",
        "yarn",
        "pnpm",
        "pip",
        "pip3",
        "pipx",
        "poetry",
        "uv",
        "pytest",
        "python",
        "python3",
        "node",
        "nodejs",
        "cargo",
        "rustc",
        "rustup",
        "go",
        "gofmt",
        "golint",
        "golangci-lint",
        "make",
        "cmake",
        "meson",
        "ninja",
        "bazel",
        "buck",
        "gradle",
        "mvn",
        "mvnw",
        "ant",
        "sbt",
        "mill",
        "dotnet",
        "nuget",
        "msbuild",
        "docker",
        "docker-compose",
        "podman",
        "buildah",
        "skopeo",
        "kubectl",
        "helm",
        "kustomize",
        "terraform",
        "tofu",
        "ansible",
        "ansible-playbook",
        "vagrant",
        "packer",
        "jq",
        "yq",
        "sed",
        "awk",
        "grep",
        "rg",
        "ag",
        "ack",
        "find",
        "fd",
        "ls",
        "exa",
        "eza",
        "tree",
        "cat",
        "bat",
        "less",
        "more",
        "head",
        "tail",
        "wc",
        "sort",
        "uniq",
        "cut",
        "tr",
        "tee",
        "xargs",
        "parallel",
        "curl",
        "wget",
        "httpie",
        "http",
        "ssh",
        "scp",
        "rsync",
        "tar",
        "gzip",
        "gunzip",
        "bzip2",
        "xz",
        "unzip",
        "7z",
        "zip",
        "vim",
        "nvim",
        "vi",
        "nano",
        "emacs",
        "code",
        "gedit",
        "subl",
        "atom",
        "clang",
        "gcc",
        "g++",
        "clang++",
        "cc",
        "c++",
        "ld",
        "ar",
        "nm",
        "objdump",
        "readelf",
        "strip",
        "valgrind",
        "gdb",
        "lldb",
        "strace",
        "ltrace",
        "perf",
        "bpftrace",
        "dtrace",
        "systemtap",
        "time",
        "/usr/bin/time",
        "bash",
        "zsh",
        "fish",
        "sh",
        "dash",
        "python",
        "python3",
        "perl",
        "ruby",
        "lua",
        "node",
        "deno",
        "bun",
    }
)

_logger = get_logger(__name__)

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
    ``||``, ``|`` and newlines."""
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
        if not _target_inside_workspace(target, workspace):
            return True
    return False


def _is_force_push(segment: str, normalized: str) -> bool:
    """Report whether a segment is a git force push."""
    if "git" not in normalized or "push" not in normalized:
        return False
    # Covers --force, --force-with-lease and --force-with-lease=<ref>.
    if "--force" in normalized or "--force-with-lease" in normalized:
        return True
    # A short -f flag only counts as a shell token, so that ordinary
    # commands (e.g. "ls -f") are not mistaken for a force push.
    return any(token.lower() == "-f" for token in _shell_tokens(segment))


def _is_dangerous_env_assignment(segment: str, normalized: str) -> bool:
    """Report whether a segment contains dangerous environment variable assignments."""
    # Block assignments that could override security-critical variables
    dangerous_vars = {
        "path",
        "ld_library_path",
        "ld_preload",
        "pythonpath",
        "perl5lib",
        "ruby_lib",
        "node_path",
        "go_path",
        "gopath",
        "cargo_home",
        "rustup_home",
        "home",
        "user",
        "logname",
        "shell",
        "term",
        "ifs",
        "ps1",
        "ps2",
        "ps4",
        "prompt",
        "editor",
        "visual",
        "pager",
        "manpath",
        "infopath",
        "classpath",
        "java_home",
        "maven_home",
        "gradle_home",
        "ant_home",
    }
    # Check for VAR=value assignments at the start of the segment
    for var in dangerous_vars:
        pattern = rf"^{var}\s*="
        if re.search(pattern, normalized, re.IGNORECASE):
            return True
    return False


def _is_dangerous_segment(segment: str, workspace: Path) -> bool:
    """Report whether a single command segment violates the command policy."""
    normalized = _normalize_command(segment)
    if not normalized:
        return False

    # ALLOW list check first - if the program is explicitly allowed, skip other checks
    program = _get_program_basename(segment)
    if program:
        program_lower = program.lower()
        if program_lower in ALLOWED_PROGRAMS:
            # Still check for force push and redirection escapes even for allowed programs
            if _is_force_push(segment, normalized):
                return True
            if _redirects_outside_workspace(segment, workspace):
                return True
            return False

    # DENY by PROGRAM NAME
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

    # DENY dangerous environment variable assignments
    if _is_dangerous_env_assignment(segment, normalized):
        return True

    # DENY a redirection that escapes the workspace
    return bool(_redirects_outside_workspace(segment, workspace))


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
    # Check for process substitution <(...), >(...)
    if re.search(r"[<>]\(.*\)", text):
        return True
    # Check for subshells (...)
    if re.search(r"\([^)]*\)", text) and not re.search(r"\$\(", text):
        # Only flag subshells that aren't command substitution
        pass  # Allow subshells for now as they're common in scripts
    # Check for here documents and here strings
    if re.search(r"<<[-<]?\s*\w+", text):
        return True
    if re.search(r"<<<\s*", text):
        return True
    return False


def is_command_denied(command: str, workspace: Path) -> bool:
    """Check if command is denied by security policy."""
    if not command or not command.strip():
        return False
    return _text_is_denied(command, workspace, depth=0)


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
        self._tool_results: list[dict[str, Any]] = []
        self._iteration_start_time: float = 0
        self._total_tool_calls: int = 0

    async def run(self) -> WorkerResult:
        set_task_id(self.task.id)
        try:
            return await self._run()
        except Exception as exc:  # pragma: no cover - defensive
            _logger.error("worker_run_failed", task_id=self.task.id, error=str(exc))
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
            self._iteration_start_time = time.time()
            _logger.debug("worker_iteration_start", task_id=self.task.id, iteration=iterations)
            response: LLMResponse = await self.client.chat(
                messages, tools=WORKER_TOOLS, system=WORKER_SYSTEM, model=self.settings.worker_model
            )
            if not response.tool_calls:
                summary = response.text.strip() if response.text.strip() else "Task completed."
                _logger.info("worker_completed", task_id=self.task.id, iterations=iterations, files_changed=len(self.files_changed))
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
                self._total_tool_calls += 1
                tool_start = time.time()
                result = await self._execute_tool(tc)
                tool_duration = time.time() - tool_start
                result_blocks.append(
                    {"type": "tool_result", "tool_use_id": tc.id, "content": result}
                )
                # Track tool execution for debugging
                self._tool_results.append({
                    "tool": tc.name,
                    "args": tc.arguments,
                    "result": result[:500] if isinstance(result, str) else str(result)[:500],
                    "duration_ms": round(tool_duration * 1000, 2),
                    "success": not isinstance(result, str) or not result.startswith("error:"),
                })
            messages.append({"role": "user", "content": result_blocks})

        # budget exhausted: the worker was truncated, so this is NOT a success.
        # Reporting success here would make the orchestrator mark the task as
        # completed instead of retrying it.
        _logger.warning("worker_iteration_budget_exhausted", task_id=self.task.id, iterations=max_iters)
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
        if self.task.dependencies:
            parts.append(f"Dependencies: {', '.join(self.task.dependencies)}")
        return "\n".join(parts)

    async def _execute_tool(self, tc: ToolCall) -> str:
        name = tc.name
        args = self._coerce_tool_args(tc.arguments)
        if args is None:
            _logger.warning("tool_invalid_args", task_id=self.task.id, tool=name, args=tc.arguments)
            return "error: invalid tool arguments"
        _logger.debug("tool_executing", task_id=self.task.id, tool=name, args=args)
        if name == "read_file":
            return await self._read_file(str(args.get("path", "")))
        if name == "write_file":
            return await self._write_file(str(args.get("path", "")), str(args.get("content", "")))
        if name == "list_files":
            return await self._list_files(
                str(args.get("path", "")) if args.get("path") is not None else "."
            )
        if name == "run_command":
            return await self._run_command(str(args.get("command", "")))
        _logger.warning("tool_unknown", task_id=self.task.id, tool=name)
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
            except (json.JSONDecodeError, ValueError):
                return None
            if isinstance(parsed, dict):
                return parsed
            return None
        return None

    async def _read_file(self, path: str) -> str:
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            _logger.warning("read_file_denied", task_id=self.task.id, path=path)
            return "error: path escapes workspace"
        try:
            content = await self.client.read_file(resolved)
        except (OSError, ValueError, RuntimeError) as exc:
            _logger.error("read_file_error", task_id=self.task.id, path=path, error=str(exc))
            return f"error: {exc}"
        if len(content) > 20000:
            content = content[:20000] + "\n[... output truncated ...]"
        return content

    async def _write_file(self, path: str, content: str) -> str:
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            _logger.warning("write_file_denied", task_id=self.task.id, path=path)
            return "error: path escapes workspace"
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            await self.client.write_file(resolved, content)
        except (OSError, ValueError, RuntimeError) as exc:
            _logger.error("write_file_error", task_id=self.task.id, path=path, error=str(exc))
            return f"error: {exc}"
        rel = str(resolved.relative_to(self.workspace.resolve()))
        self.files_changed.add(rel)
        _logger.info("file_written", task_id=self.task.id, path=path, size=len(content.encode('utf-8')))
        return f"wrote {path} ({len(content.encode('utf-8'))} bytes)"

    async def _list_files(self, path: str) -> str:
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            _logger.warning("list_files_denied", task_id=self.task.id, path=path)
            return "error: path escapes workspace"
        try:
            entries = await asyncio.to_thread(self.client.list_files, resolved)
        except (OSError, ValueError, RuntimeError) as exc:
            _logger.error("list_files_error", task_id=self.task.id, path=path, error=str(exc))
            return f"error: {exc}"
        # Cap the listing at MAX_LIST_ENTRIES so huge trees cannot flood the context.
        rels = []
        base = self.workspace.resolve()
        for e in entries[:MAX_LIST_ENTRIES]:
            try:
                rels.append(str(Path(e).relative_to(base)) if Path(e).is_absolute() else e)
            except ValueError:
                rels.append(str(e))
        _logger.debug("list_files_done", task_id=self.task.id, path=path, count=len(rels))
        return "\n".join(rels)

    async def _run_command(self, command: str) -> str:
        if not command.strip():
            return "error: empty command"
        # The hardened check subsumes the legacy pattern list: it normalizes
        # the command, checks every segment against DENIED_COMMANDS and
        # DENIED_PROGRAMS, and denies force pushes and redirections outside
        # the workspace.
        if is_command_denied(command, self.workspace):
            _logger.warning("command_denied", task_id=self.task.id, command=command[:200])
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
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **spawn_kwargs,
                )
        except (OSError, ValueError, RuntimeError) as exc:
            _logger.error("command_spawn_failed", task_id=self.task.id, command=command[:200], error=str(exc))
            return f"error: {exc}"
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            _logger.warning("command_timeout", task_id=self.task.id, command=command[:200], timeout=timeout)
            await self._kill_process_tree(proc, new_session)
            return "command timed out"
        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        combined = out + "\n[stderr]\n" + err if err.strip() else out
        cap = 20000
        if len(combined) > cap:
            combined = "[... output truncated ...]\n" + combined[-cap:]
        _logger.debug("command_completed", task_id=self.task.id, command=command[:100], returncode=proc.returncode)
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
