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
from furrow.logging import get_logger

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
    # Additional dangerous patterns
    "curl | sh",
    "curl | bash",
    "wget | sh",
    "wget | bash",
    "curl -fsSL | sh",
    "curl -fsSL | bash",
    "base64 -d | sh",
    "base64 -d | bash",
    "eval $",
    "eval \"$",
    "eval '$",
    "source <(",
    ". <(",
    "process substitution",
    "/dev/tcp/",
    "/dev/udp/",
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
        # Additional dangerous programs
        "fdisk",
        "cfdisk",
        "sfdisk",
        "gdisk",
        "partprobe",
        "kpartx",
        "losetup",
        "cryptsetup",
        "lvm",
        "vgcreate",
        "lvcreate",
        "mkfs.ext2",
        "mkfs.ext3",
        "mkfs.ext4",
        "mkfs.xfs",
        "mkfs.btrfs",
        "mkfs.vfat",
        "mkfs.ntfs",
        "mkswap",
        "swapon",
        "swapoff",
        "mount",
        "umount",
        "fsck",
        "e2fsck",
        "xfs_repair",
        "btrfsck",
    }
)

# Patterns that indicate potential command injection or obfuscation
SUSPICIOUS_PATTERNS = (
    r"base64\s+-d\s*\|\s*(sh|bash)",
    r"(curl|wget)\s+.*\s*\|\s*(sh|bash)",
    r"eval\s+\$\{",
    r"eval\s+\$\([^)]+\)",
    r"source\s+<\(",
    r"\.\s+<\(",
    r"/dev/(tcp|udp)/",
    r"bash\s+-c\s+.*base64",
    r"sh\s+-c\s+.*base64",
    r"python\s+-c\s+.*base64",
    r"perl\s+-e\s+.*base64",
)

# Maximum number of entries returned by the ``list_files`` tool.
MAX_LIST_ENTRIES = 500

# Maximum number of matches returned by the ``search_files`` tool.
MAX_SEARCH_MATCHES = 200

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


def _is_dangerous_segment(segment: str, workspace: Path) -> bool:
    """Report whether a single command segment violates the command policy."""
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

    # DENY by SUSPICIOUS PATTERNS (command injection/obfuscation)
    import re
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return True

    # DENY a force push
    if _is_force_push(segment, normalized):
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
    # Check suspicious patterns on the whole text
    import re
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
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
        self._logger = get_logger("furrow.worker").bind(task_id=task.id, task_desc=task.description[:100])

    async def run(self) -> WorkerResult:
        self._logger.info("worker_started", max_iterations=self.settings.max_worker_iterations)
        try:
            result = await self._run()
            self._logger.info("worker_finished", success=result.success, iterations=result.iterations,
                             files_changed=result.files_changed)
            return result
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.error("worker_exception", error=str(exc))
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
            self._logger.debug("worker_iteration", iteration=iterations)
            response: LLMResponse = await self.client.chat(
                messages, tools=WORKER_TOOLS, system=WORKER_SYSTEM, model=self.settings.worker_model, agent="worker"
            )
            if not response.tool_calls:
                summary = response.text.strip() if response.text.strip() else "Task completed."
                self._logger.info("worker_completed", summary=summary[:200])
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
                self._logger.debug("tool_call", tool=tc.name, args=tc.arguments)
                result = await self._execute_tool(tc)
                self._logger.debug("tool_result", tool=tc.name, result_preview=result[:200] if isinstance(result, str) else str(result)[:200])
                result_blocks.append(
                    {"type": "tool_result", "tool_use_id": tc.id, "content": result}
                )
            messages.append({"role": "user", "content": result_blocks})

        # budget exhausted: the worker was truncated, so this is NOT a success.
        # Reporting success here would make the orchestrator mark the task as
        # completed instead of retrying it.
        self._logger.warning("worker_budget_exhausted", max_iterations=max_iters)
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
            return await self._write_file(str(args.get("path", "")), str(args.get("content", "")))
        if name == "patch_file":
            return await self._patch_file(str(args.get("path", "")), str(args.get("patch", "")))
        if name == "search_files":
            return await self._search_files(
                str(args.get("pattern", "")),
                str(args.get("path", "")) if args.get("path") is not None else ".",
                str(args.get("include", "")) if args.get("include") is not None else "",
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
            except (json.JSONDecodeError, ValueError):
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

    async def _patch_file(self, path: str, patch: str) -> str:
        """Apply a unified diff patch to a file."""
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            return "error: path escapes workspace"
        if not patch.strip():
            return "error: empty patch"
        try:
            # Read the original file
            original = await self.client.read_file(resolved)
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: could not read file: {exc}"
        # Apply the patch
        try:
            patched = self._apply_unified_diff(original, patch)
        except Exception as exc:
            return f"error: patch failed: {exc}"
        # Write the patched content
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            await self.client.write_file(resolved, patched)
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"
        rel = str(resolved.relative_to(self.workspace.resolve()))
        self.files_changed.add(rel)
        return f"patched {path} ({len(patched.encode('utf-8'))} bytes)"

    @staticmethod
    def _apply_unified_diff(original: str, patch: str) -> str:
        """Apply a unified diff to the original content. Returns the patched content."""
        import difflib
        
        original_lines = original.splitlines(keepends=True)
        patch_lines = patch.splitlines(keepends=True)
        
        # Parse the unified diff header
        if len(patch_lines) < 3:
            raise ValueError("Invalid patch format: too short")
        
        # Skip the header lines (--- and +++)
        i = 0
        while i < len(patch_lines) and (patch_lines[i].startswith("---") or patch_lines[i].startswith("+++")):
            i += 1
        
        # Parse hunks
        result_lines = original_lines[:]
        line_offset = 0
        
        while i < len(patch_lines):
            line = patch_lines[i]
            if not line.startswith("@@"):
                i += 1
                continue
            
            # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            import re
            hunk_match = re.match(r"^@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@", line)
            if not hunk_match:
                raise ValueError(f"Invalid hunk header: {line}")
            
            old_start = int(hunk_match.group(1)) - 1  # Convert to 0-indexed
            old_count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
            new_start = int(hunk_match.group(3)) - 1  # Convert to 0-indexed
            new_count = int(hunk_match.group(4)) if hunk_match.group(4) else 1
            
            i += 1
            hunk_lines = []
            while i < len(patch_lines) and not patch_lines[i].startswith("@@"):
                hunk_lines.append(patch_lines[i])
                i += 1
            
            # Apply the hunk
            # Find the matching context in the original
            context_before = []
            context_after = []
            changes = []
            
            for hline in hunk_lines:
                if hline.startswith(" "):
                    context_before.append(hline[1:])
                elif hline.startswith("-"):
                    context_before.append(hline[1:])
                    changes.append(("delete", hline[1:]))
                elif hline.startswith("+"):
                    changes.append(("add", hline[1:]))
            
            # Simple approach: replace the old lines with new lines
            # Build the new lines for this hunk
            new_hunk_lines = []
            for hline in hunk_lines:
                if hline.startswith(" "):
                    new_hunk_lines.append(hline[1:])
                elif hline.startswith("+"):
                    new_hunk_lines.append(hline[1:])
                # Skip deleted lines
            
            # Replace in result_lines
            start_idx = old_start + line_offset
            end_idx = start_idx + old_count
            
            # Validate the context matches
            if start_idx < 0 or end_idx > len(result_lines):
                raise ValueError(f"Hunk range out of bounds: {start_idx}-{end_idx} (file has {len(result_lines)} lines)")
            
            # Check context matches (first and last context lines)
            if old_count > 0:
                # Verify first context line if it exists
                if hunk_lines and hunk_lines[0].startswith(" "):
                    expected = hunk_lines[0][1:]
                    actual = result_lines[start_idx] if start_idx < len(result_lines) else ""
                    if expected != actual:
                        raise ValueError(f"Context mismatch at line {start_idx + 1}: expected {expected!r}, got {actual!r}")
            
            # Apply the change
            result_lines[start_idx:end_idx] = new_hunk_lines
            line_offset += len(new_hunk_lines) - old_count
        
        return "".join(result_lines)

    async def _search_files(self, pattern: str, path: str, include: str) -> str:
        """Search for a pattern in files under the workspace."""
        if not pattern.strip():
            return "error: empty pattern"
        
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            return "error: path escapes workspace"
        
        try:
            # Use grep-like search with ripgrep if available, otherwise Python fallback
            import subprocess
            cmd = ["rg", "--no-heading", "--line-number", "--color=never"]
            if include:
                cmd.extend(["-g", include])
            cmd.extend([pattern, str(resolved)])
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            
            if proc.returncode == 0:
                output = stdout.decode("utf-8", errors="replace")
            elif proc.returncode == 1:
                # No matches found
                output = ""
            else:
                # Try Python fallback
                output = await self._search_files_python(pattern, resolved, include)
        except (FileNotFoundError, OSError, ValueError, RuntimeError, asyncio.TimeoutError):
            # rg not available or error, use Python fallback
            output = await self._search_files_python(pattern, resolved, include)
        
        # Cap the output
        lines = output.splitlines()
        if len(lines) > MAX_SEARCH_MATCHES:
            lines = lines[:MAX_SEARCH_MATCHES]
            lines.append(f"... ({len(output.splitlines()) - MAX_SEARCH_MATCHES} more matches truncated)")
        return "\n".join(lines)

    async def _search_files_python(self, pattern: str, resolved: Path, include: str) -> str:
        """Python fallback for file searching."""
        import re
        regex = re.compile(pattern)
        results = []
        base = self.workspace.resolve()
        
        # Walk the directory
        for file_path in resolved.rglob("*"):
            if not file_path.is_file():
                continue
            # Check include pattern
            if include:
                import fnmatch
                rel = file_path.relative_to(base)
                if not fnmatch.fnmatch(str(rel), include):
                    continue
            try:
                content = await self.client.read_file(file_path)
            except (OSError, ValueError, RuntimeError):
                continue
            for i, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    rel = str(file_path.relative_to(base))
                    results.append(f"{rel}:{i}:{line}")
                    if len(results) >= MAX_SEARCH_MATCHES:
                        break
            if len(results) >= MAX_SEARCH_MATCHES:
                break
        return "\n".join(results)

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
