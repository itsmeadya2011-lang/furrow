from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.workspace = self.settings.workspace

    async def run(self) -> str:
        # First, read any relevant files to understand the codebase
        context = await self._gather_context()

        prompt = f"""{WORKER_PROMPT}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any relevant files'}

Context from codebase:
{context}

You can use the following tools by responding with JSON:
- {{"tool": "read", "path": "relative/path.py"}} - Read a file
- {{"tool": "write", "path": "relative/path.py", "content": "file content"}} - Write/create a file
- {{"tool": "edit", "path": "relative/path.py", "old": "old text", "new": "new text"}} - Edit a file (exact match)
- {{"tool": "list", "path": "directory"}} - List files in directory
- {{"tool": "done", "summary": "Task completed: ..."}} - Mark task as complete

Return ONLY a JSON object with the tool call. Chain multiple tool calls by returning them one at a time.
When the task is complete, return {{"tool": "done", "summary": "..."}}.
"""
        max_rounds = 10
        for round_num in range(max_rounds):
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
            try:
                action = json.loads(response.strip())
            except json.JSONDecodeError:
                # If not valid JSON, treat as final summary
                return f"Worker response (non-JSON): {response[:500]}"

            tool = action.get("tool")
            if tool == "read":
                result = await self._read_file(action["path"])
                prompt = f"File content:\n{result}\n\nNext action:"
            elif tool == "write":
                result = await self._write_file(action["path"], action["content"])
                prompt = f"{result}\n\nNext action:"
            elif tool == "edit":
                result = await self._edit_file(action["path"], action["old"], action["new"])
                prompt = f"{result}\n\nNext action:"
            elif tool == "list":
                result = self._list_files(action.get("path", "."))
                prompt = f"Files:\n{result}\n\nNext action:"
            elif tool == "done":
                return action.get("summary", "Task completed")
            else:
                prompt = f"Unknown tool: {tool}. Available: read, write, edit, list, done\n\nNext action:"

        return f"Worker exceeded max rounds ({max_rounds}). Last action: {response[:200]}"

    async def _gather_context(self) -> str:
        """Read relevant files to provide context to the worker."""
        context_parts = []
        files_to_read = self.task.files[:5]  # Limit to first 5 files

        for file_path in files_to_read:
            full_path = self.workspace / file_path
            if full_path.exists():
                try:
                    content = await self.client.read_file(full_path)
                    context_parts.append(f"--- {file_path} ---\n{content[:2000]}")
                except Exception as e:
                    context_parts.append(f"--- {file_path} ---\nError reading: {e}")

        if not context_parts and self.workspace.exists():
            # List workspace files as fallback
            files = self.client.list_files(self.workspace)
            context_parts.append(f"Workspace files: {', '.join(files[:30])}")

        return "\n\n".join(context_parts) if context_parts else "No existing files found."

    async def _read_file(self, path: str) -> str:
        full_path = self.workspace / path
        try:
            content = await self.client.read_file(full_path)
            return content
        except Exception as e:
            return f"Error reading {path}: {e}"

    async def _write_file(self, path: str, content: str) -> str:
        full_path = self.workspace / path
        try:
            await self.client.write_file(full_path, content)
            return f"Successfully wrote {path}"
        except Exception as e:
            return f"Error writing {path}: {e}"

    async def _edit_file(self, path: str, old: str, new: str) -> str:
        full_path = self.workspace / path
        try:
            content = await self.client.read_file(full_path)
            if old not in content:
                return f"Error: Could not find exact match for edit in {path}. Old text not found."
            new_content = content.replace(old, new, 1)
            await self.client.write_file(full_path, new_content)
            return f"Successfully edited {path}"
        except Exception as e:
            return f"Error editing {path}: {e}"

    def _list_files(self, path: str) -> str:
        full_path = self.workspace / path
        files = self.client.list_files(full_path)
        return "\n".join(f"  {f}" for f in files) if files else "  (empty)"