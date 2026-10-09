from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        # Build the prompt with context about files to touch
        files_context = ", ".join(self.task.files) if self.task.files else "any file in the workspace"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_context}\n\n"
            f"Current workspace contents:\n{self._list_workspace_files()}"
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Try to parse file changes from the response
        # The LLM is expected to return JSON with file changes
        changes = self._parse_file_changes(response)
        if changes:
            for file_path, content in changes.items():
                await self.client.write_file(file_path, content)
            return f"Updated {len(changes)} file(s): {', '.join(sorted(changes.keys()))}"

        # Fallback: if no structured changes found, return the raw response
        return response

    def _list_workspace_files(self) -> str:
        """List files in the workspace to give the worker context."""
        try:
            files = self.client.list_files(self.client.settings.workspace)
            if not files:
                return "(empty workspace)"
            # Show up to 50 files
            return "\n".join(sorted(files)[:50])
        except Exception:
            return "(unable to list files)"

    def _parse_file_changes(self, response: str) -> dict[str, str]:
        """Parse file changes from LLM response.

        Expects JSON like:
        {
            "files": {
                "path/to/file.py": "file content"
            }
        }
        """
        # Try to find JSON in the response
        response = response.strip()

        # Try direct JSON parse
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "files" in data:
                return data["files"]
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict) and "files" in data:
                    return data["files"]
            except (json.JSONDecodeError, TypeError):
                pass

        return {}