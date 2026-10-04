from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, ClassVar

import aiofiles

from furrow.config import Settings, settings
from furrow.errors import ToolError, WorkspaceContainmentError


@dataclass(slots=True)
class ToolResult:
    """Outcome of a single tool invocation."""
    name: str
    ok: bool
    output: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
        }


Handler = Callable[[dict[str, Any]], Awaitable[ToolResult]]

_MAX_LIST_ENTRIES = 500


class WorkspaceTools:
    """Sandboxed filesystem and command tools rooted at a workspace directory.

    Every path is resolved and confirmed to be inside the workspace root
    before it is touched; a path that escapes raises
    WorkspaceContainmentError. Commands are refused unless `allow_shell` is
    set.
    """

    _SKIP_DIRS: ClassVar[set[str]] = {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
    }
    _SKIP_PATTERNS: ClassVar[tuple[str, ...]] = ("*.egg-info",)

    def __init__(
        self,
        workspace: str | Path,
        *,
        allow_shell: bool = False,
        shell_timeout: int = 60,
        max_output_chars: int = 20_000,
    ) -> None:
        self._root = Path(workspace).resolve()
        self._allow_shell = allow_shell
        self._shell_timeout = shell_timeout
        self._max_output_chars = max_output_chars

        self._register: dict[str, Handler] = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "edit_file": self.edit_file,
            "list_files": self.list_files,
            "run_command": self.run_command,
        }

    @classmethod
    def from_settings(cls, cfg: Settings | None = None) -> WorkspaceTools:
        """Build from Settings, defaulting to the module-level `settings`
        singleton when cfg is None."""
        if cfg is None:
            cfg = settings
        return cls(
            workspace=cfg.workspace,
            allow_shell=cfg.allow_shell,
            shell_timeout=cfg.shell_timeout,
        )

    @property
    def root(self) -> Path:
        return self._root

    def _resolve(self, path: str) -> Path:
        """Resolve a path and verify it is contained within the workspace root."""
        p = Path(path).expanduser()
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (self._root / p).resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError as e:
            raise WorkspaceContainmentError(f"Path escapes workspace: {path}") from e
        return resolved

    def _truncate(self, text: str) -> str:
        """Truncate text to max_output_chars and append notice if truncated."""
        if len(text) <= self._max_output_chars:
            return text
        truncated = text[: self._max_output_chars]
        notice = f"\n... [truncated, {len(text) - self._max_output_chars} more characters]"
        return truncated + notice

    # --- provider-agnostic schema ---
    def specs(self) -> list[dict[str, Any]]:
        """Provider-neutral tool definitions:
        [{"name": str, "description": str, "input_schema": {...JSON Schema...}}]
        `input_schema` must be a valid JSON Schema object with "type": "object",
        "properties", and "required"."""
        return [
            {
                "name": "read_file",
                "description": "Read the contents of a file within the workspace.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file to read.",
                        }
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": "Write content to a file within the workspace, creating parent directories as needed.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file to write.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file.",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "edit_file",
                "description": "Make a surgical edit to a file by replacing occurrences of a string.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file to edit.",
                        },
                        "old": {
                            "type": "string",
                            "description": "String to replace. Must exist in the file.",
                        },
                        "new": {
                            "type": "string",
                            "description": "String to replace with.",
                        },
                        "replace_all": {
                            "type": "boolean",
                            "description": "If true, replace all occurrences. If false (default), require exactly one occurrence.",
                            "default": False,
                        },
                    },
                    "required": ["path", "old", "new"],
                },
            },
            {
                "name": "list_files",
                "description": "List files recursively within a directory, skipping common build/cache directories.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the directory to list. Defaults to workspace root.",
                            "default": ".",
                        }
                    },
                    "required": [],
                },
            },
            {
                "name": "run_command",
                "description": "Run a shell command within the workspace. Requires allow_shell=True.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute.",
                        }
                    },
                    "required": ["command"],
                },
            },
        ]

    def anthropic_tools(self) -> list[dict[str, Any]]:
        """Anthropic Messages API shape: [{"name", "description",
        "input_schema"}] — i.e. identical to specs()."""
        return self.specs()

    def openai_tools(self) -> list[dict[str, Any]]:
        """OpenAI chat-completions shape:
        [{"type": "function", "function": {"name", "description",
        "parameters"}}]."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in self.specs()
        ]

    def names(self) -> list[str]:
        return list(self._register.keys())

    async def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Dispatch to a tool. Unknown name -> ToolResult(ok=False,
        error=...). Never raises for ordinary failures; always returns a
        ToolResult."""
        handler = self._register.get(name)
        if handler is None:
            return ToolResult(
                name=name,
                ok=False,
                error=f"Unknown tool: {name}",
            )
        try:
            return await handler(arguments)
        except WorkspaceContainmentError as e:
            return ToolResult(name=name, ok=False, error=str(e))
        except ToolError as e:
            return ToolResult(name=name, ok=False, error=str(e))
        # Broad catch on purpose: a tool sandbox must never crash the loop.
        except Exception as e:
            return ToolResult(name=name, ok=False, error=f"Unexpected error: {e}")

    # --- individual tools ---
    async def read_file(self, arguments: dict[str, Any]) -> ToolResult:
        path_str = arguments.get("path")
        if not isinstance(path_str, str):
            return ToolResult(name="read_file", ok=False, error="path must be a string")
        try:
            resolved = self._resolve(path_str)
        except WorkspaceContainmentError as e:
            return ToolResult(name="read_file", ok=False, error=str(e))

        try:
            async with aiofiles.open(resolved, encoding="utf-8") as f:
                content = await f.read()
        except FileNotFoundError:
            return ToolResult(name="read_file", ok=False, error=f"File not found: {path_str}")
        except IsADirectoryError:
            return ToolResult(name="read_file", ok=False, error=f"Path is a directory: {path_str}")
        # Broad catch on purpose: a tool sandbox must never crash the loop.
        except Exception as e:
            return ToolResult(name="read_file", ok=False, error=f"Failed to read file: {e}")

        output = self._truncate(content)
        return ToolResult(name="read_file", ok=True, output=output)

    async def write_file(self, arguments: dict[str, Any]) -> ToolResult:
        path_str = arguments.get("path")
        content = arguments.get("content")
        if not isinstance(path_str, str):
            return ToolResult(name="write_file", ok=False, error="path must be a string")
        if not isinstance(content, str):
            return ToolResult(name="write_file", ok=False, error="content must be a string")

        try:
            resolved = self._resolve(path_str)
        except WorkspaceContainmentError as e:
            return ToolResult(name="write_file", ok=False, error=str(e))

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write: write to temp file then replace
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=resolved.parent,
                delete=False,
                prefix=f".{resolved.name}.tmp.",
            ) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, resolved)
        # Broad catch on purpose: a tool sandbox must never crash the loop.
        except Exception as e:
            return ToolResult(name="write_file", ok=False, error=f"Failed to write file: {e}")

        byte_count = len(content.encode("utf-8"))
        return ToolResult(
            name="write_file",
            ok=True,
            output=f"Wrote {byte_count} bytes to {path_str}",
        )

    async def edit_file(self, arguments: dict[str, Any]) -> ToolResult:
        path_str = arguments.get("path")
        old = arguments.get("old")
        new = arguments.get("new")
        replace_all = arguments.get("replace_all", False)

        if not isinstance(path_str, str):
            return ToolResult(name="edit_file", ok=False, error="path must be a string")
        if not isinstance(old, str):
            return ToolResult(name="edit_file", ok=False, error="old must be a string")
        if not isinstance(new, str):
            return ToolResult(name="edit_file", ok=False, error="new must be a string")
        if not isinstance(replace_all, bool):
            return ToolResult(name="edit_file", ok=False, error="replace_all must be a boolean")

        try:
            resolved = self._resolve(path_str)
        except WorkspaceContainmentError as e:
            return ToolResult(name="edit_file", ok=False, error=str(e))

        try:
            async with aiofiles.open(resolved, encoding="utf-8") as f:
                content = await f.read()
        except FileNotFoundError:
            return ToolResult(name="edit_file", ok=False, error=f"File not found: {path_str}")
        except IsADirectoryError:
            return ToolResult(name="edit_file", ok=False, error=f"Path is a directory: {path_str}")
        # Broad catch on purpose: a tool sandbox must never crash the loop.
        except Exception as e:
            return ToolResult(name="edit_file", ok=False, error=f"Failed to read file: {e}")

        count = content.count(old)
        if count == 0:
            return ToolResult(name="edit_file", ok=False, error="old_string not found")
        if count > 1 and not replace_all:
            return ToolResult(
                name="edit_file",
                ok=False,
                error=f"old_string not unique, {count} occurrences; pass replace_all or add context",
            )

        new_content = content.replace(old, new)
        if new_content == content:
            return ToolResult(name="edit_file", ok=False, error="no changes made")

        try:
            # Atomic write
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=resolved.parent,
                delete=False,
                prefix=f".{resolved.name}.tmp.",
            ) as tmp:
                tmp.write(new_content)
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, resolved)
        # Broad catch on purpose: a tool sandbox must never crash the loop.
        except Exception as e:
            return ToolResult(name="edit_file", ok=False, error=f"Failed to write file: {e}")

        return ToolResult(
            name="edit_file",
            ok=True,
            output=f"Made {count} replacement(s) in {path_str}",
        )

    async def list_files(self, arguments: dict[str, Any]) -> ToolResult:
        path_str = arguments.get("path", ".")
        if not isinstance(path_str, str):
            return ToolResult(name="list_files", ok=False, error="path must be a string")

        try:
            resolved = self._resolve(path_str)
        except WorkspaceContainmentError as e:
            return ToolResult(name="list_files", ok=False, error=str(e))

        if not resolved.exists():
            return ToolResult(name="list_files", ok=False, error=f"Directory not found: {path_str}")
        if not resolved.is_dir():
            return ToolResult(name="list_files", ok=False, error=f"Path is not a directory: {path_str}")

        entries: list[str] = []
        try:
            for entry in resolved.rglob("*"):
                # Skip anything under a build/cache directory.
                rel = entry.relative_to(resolved)
                if any(
                    part in self._SKIP_DIRS
                    or fnmatch(part, pattern)
                    for part in rel.parts
                    for pattern in self._SKIP_PATTERNS
                ):
                    continue
                if entry.is_file():
                    entries.append(str(rel))
                elif entry.is_dir():
                    entries.append(str(rel) + "/")
        # Broad catch on purpose: a tool sandbox must never crash the loop.
        except Exception as e:
            return ToolResult(name="list_files", ok=False, error=f"Failed to list files: {e}")

        entries.sort()
        total = len(entries)
        if total > _MAX_LIST_ENTRIES:
            entries = entries[:_MAX_LIST_ENTRIES]
            notice = f"\n... [truncated, showing first {_MAX_LIST_ENTRIES} of {total} entries]"
        else:
            notice = ""

        output = "\n".join(entries) + notice
        return ToolResult(name="list_files", ok=True, output=output)

    async def run_command(self, arguments: dict[str, Any]) -> ToolResult:
        if not self._allow_shell:
            return ToolResult(
                name="run_command",
                ok=False,
                error="Shell access is disabled. Set allow_shell=True to enable.",
            )

        command = arguments.get("command")
        if not isinstance(command, str):
            return ToolResult(name="run_command", ok=False, error="command must be a string")

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=self._root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._shell_timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                # Drain the pipes so the subprocess transport closes cleanly.
                await proc.communicate()
                return ToolResult(
                    name="run_command",
                    ok=False,
                    error=f"Command timed out after {self._shell_timeout}s",
                )

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")
            combined = stdout_str + stderr_str
            output = self._truncate(combined)
            output += f"\n[exit code: {proc.returncode}]"

            ok = proc.returncode == 0
            return ToolResult(name="run_command", ok=ok, output=output)
        # Broad catch on purpose: a tool sandbox must never crash the loop.
        except Exception as e:
            return ToolResult(name="run_command", ok=False, error=f"Failed to run command: {e}")
