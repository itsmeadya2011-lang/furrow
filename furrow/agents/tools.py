from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.config import Provider

if TYPE_CHECKING:
    from furrow.config import Settings


MAX_FILE_BYTES = 1_000_000
MAX_BASH_OUTPUT = 8000


class Tool(ABC):
    """Abstract base class for agent tools."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def params(self) -> dict: ...

    @abstractmethod
    async def run(self, **kwargs: Any) -> str: ...

    def _resolve(self, path: str | Path) -> Path:
        p = Path(path)
        if not p.is_absolute():
            base = self.settings.workspace if self.settings else Path.cwd()
            p = base / p
        return p.resolve()


# ----------------------------------------------------------------- read_file
class ReadFileTool(Tool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the full contents of a UTF-8 text file from the workspace."

    @property
    def params(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to workspace or absolute."},
                "max_bytes": {
                    "type": "integer",
                    "description": "Optional cap on bytes read (defaults to 1MB).",
                },
            },
            "required": ["path"],
        }

    async def run(self, *, path: str, max_bytes: int | None = None, **_: Any) -> str:
        p = self._resolve(path)
        if not p.exists():
            return f"Error: file not found: {path}"
        if not p.is_file():
            return f"Error: not a file: {path}"
        try:
            data = p.read_bytes()
        except OSError as e:
            return f"Error reading {path}: {e}"
        if max_bytes is not None:
            data = data[:max_bytes]
        else:
            data = data[:MAX_FILE_BYTES]
        try:
            return data.decode("utf-8", errors="replace")
        except Exception as e:  # pragma: no cover - defensive
            return f"Error decoding {path}: {e}"


# ----------------------------------------------------------------- write_file
class WriteFileTool(Tool):
    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file (creating parent directories as needed), overwriting any existing content."

    @property
    def params(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }

    async def run(self, *, path: str, content: str, **_: Any) -> str:
        p = self._resolve(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except OSError as e:
            return f"Error writing {path}: {e}"
        return f"Wrote {len(content)} bytes to {p}"


# ----------------------------------------------------------------- str_replace
class StrReplaceTool(Tool):
    @property
    def name(self) -> str:
        return "str_replace"

    @property
    def description(self) -> str:
        return "Replace a single occurrence of old_str with new_str in a file."

    @property
    def params(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
            },
            "required": ["path", "old_str", "new_str"],
        }

    async def run(self, *, path: str, old_str: str, new_str: str, **_: Any) -> str:
        p = self._resolve(path)
        if not p.exists():
            return f"Error: file not found: {path}"
        try:
            content = p.read_text(encoding="utf-8")
        except OSError as e:
            return f"Error reading {path}: {e}"
        if old_str not in content:
            return f"Error: old_str not found in {path}"
        new_content = content.replace(old_str, new_str, 1)
        try:
            p.write_text(new_content, encoding="utf-8")
        except OSError as e:
            return f"Error writing {path}: {e}"
        return f"Replaced 1 occurrence(s) in {path}"


# ----------------------------------------------------------------- list_files
class ListFilesTool(Tool):
    @property
    def name(self) -> str:
        return "list_files"

    @property
    def description(self) -> str:
        return "List files under a directory (relative to the workspace)."

    @property
    def params(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list; defaults to workspace."},
            },
        }

    async def run(self, *, path: str | None = None, **_: Any) -> str:
        base = self.settings.workspace if self.settings else Path.cwd()
        p = self._resolve(path) if path else base
        if not p.exists() or not p.is_dir():
            return f"Error: not a directory: {p}"
        try:
            entries = sorted(p.rglob("*"))
        except OSError as e:
            return f"Error listing {p}: {e}"
        lines = []
        for entry in entries:
            try:
                rel = entry.relative_to(base)
            except ValueError:
                rel = entry
            kind = "/" if entry.is_dir() else ""
            lines.append(f"{rel}{kind}")
        return "\n".join(lines) if lines else "(empty)"


# ----------------------------------------------------------------- grep
class GrepTool(Tool):
    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return "Search files for a regex pattern. Returns 'relpath:lineno:line' matches."

    @property
    def params(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "Root directory to search."},
                "glob": {"type": "string", "description": "Optional filename glob filter."},
            },
            "required": ["pattern"],
        }

    async def run(self, *, pattern: str, path: str = ".", glob: str | None = None, **_: Any) -> str:
        base = self.settings.workspace if self.settings else Path.cwd()
        root = self._resolve(path) if path else base
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: invalid regex: {e}"
        matches: list[str] = []
        try:
            iterator = root.rglob("*")
        except OSError as e:
            return f"Error walking {root}: {e}"
        for file in iterator:
            if not file.is_file():
                continue
            rel = file.relative_to(base) if base in file.parents or file == base else file
            rel_str = str(rel)
            if ".git" in rel_str.split("/"):
                continue
            if glob and not file.match(glob):
                continue
            try:
                if file.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            try:
                with file.open("r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        if regex.search(line):
                            matches.append(f"{rel_str}:{lineno}:{line.rstrip()}")
                            if len(matches) >= 500:
                                return "\n".join(matches) + "\n(truncated)"
            except (OSError, UnicodeError):
                continue
        return "\n".join(matches) if matches else "(no matches)"


# ----------------------------------------------------------------- bash
class BashTool(Tool):
    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return "Run a shell command via `bash -lc` and return combined stdout/stderr (truncated to 8000 chars)."

    @property
    def params(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "cwd": {"type": "string", "description": "Override working directory."},
                "timeout": {"type": "integer", "description": "Timeout in seconds."},
            },
            "required": ["command"],
        }

    async def run(self, *, command: str, cwd: str | None = None, timeout: int | None = None, **_: Any) -> str:
        workdir: Path | None = None
        if cwd:
            workdir = self._resolve(cwd)
        elif self.settings is not None:
            workdir = self.settings.workspace
        if workdir is not None and not workdir.exists():
            return f"Error: cwd does not exist: {workdir}"
        effective_timeout = timeout if timeout is not None else (self.settings.tool_timeout_seconds if self.settings else 60)
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-lc",
                command,
                cwd=str(workdir) if workdir else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError as e:
            return f"Error: bash not available: {e}"
        except OSError as e:
            return f"Error spawning bash: {e}"
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return f"Error: command timed out after {effective_timeout}s"
        output = stdout.decode("utf-8", errors="replace")
        if len(output) > MAX_BASH_OUTPUT:
            output = output[:MAX_BASH_OUTPUT] + f"\n(truncated at {MAX_BASH_OUTPUT} chars)"
        return output


# ----------------------------------------------------------------- registry
class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self.tools: dict[str, Tool] = {tool.name: tool for tool in tools}

    async def execute(self, name: str, args: dict) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            return await tool.run(**(args or {}))
        except TypeError as e:
            return f"Error: invalid arguments for {name}: {e}"
        except Exception as e:  # pragma: no cover - defensive
            return f"Error executing {name}: {e}"

    def schemas(self, provider: Provider) -> list[dict]:
        """Return tool schemas in the normalized form expected by LLMClient.chat."""
        return [
            {"name": tool.name, "description": tool.description, "params": tool.params}
            for tool in self.tools.values()
        ]


def default_tools(settings: Settings | None = None) -> ToolRegistry:
    return ToolRegistry(
        [
            ReadFileTool(settings),
            WriteFileTool(settings),
            StrReplaceTool(settings),
            ListFilesTool(settings),
            GrepTool(settings),
            BashTool(settings),
        ]
    )
