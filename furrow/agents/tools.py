"""File tools for the worker agent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from furrow.agents.security import MAX_LIST_ENTRIES, resolve_in_workspace
from furrow.config import Settings
from furrow.llm import LLMClient


class FileTools:
    """File operations for the worker agent."""

    def __init__(
        self,
        client: LLMClient,
        settings: Settings,
        workspace: Path,
        files_changed: set[str],
    ) -> None:
        self.client = client
        self.settings = settings
        self.workspace = workspace
        self.files_changed = files_changed

    async def read_file(self, path: str) -> str:
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

    async def write_file(self, path: str, content: str) -> str:
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

    async def list_files(self, path: str) -> str:
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


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return the tool definitions for the worker agent."""
    return [
        {
            "name": "read_file",
            "description": "read a UTF-8 text file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "create or overwrite a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "list_files",
            "description": "list files under a workspace directory (relative paths).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": [],
            },
        },
    ]