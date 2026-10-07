from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, Settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or Settings()

    async def run(self) -> str:
        # Read relevant files first to provide context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except Exception:
                file_contents[file_path] = f"[File not found: {file_path}]"

        # Build context from existing files
        context = ""
        if file_contents:
            context = "\n\nExisting files:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to modify: {', '.join(self.task.files) if self.task.files else 'any'}{context}\n\nReturn JSON with your changes:\n{{\n  \"changes\": [\n    {{\n      \"file\": \"path/to/file.py\",\n      \"content\": \"full file content after changes\"\n    }}\n  ],\n  \"summary\": \"What you changed\"\n}}"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "No summary provided")

            # Apply changes
            for change in changes:
                file_path = change.get("file")
                content = change.get("content", "")
                if file_path:
                    await self.client.write_file(file_path, content)

            return summary
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to extract code blocks from response
            return await self._fallback_execute(response)

    async def _fallback_execute(self, response: str) -> str:
        """Try to extract and apply code from markdown code blocks in the response."""
        import re

        # Look for code blocks with file paths
        pattern = r"```(\w+)?\n(?:#\s*(.+?)\n)?(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        applied = False
        for lang, file_path, content in matches:
            if file_path and file_path.strip():
                file_path = file_path.strip().lstrip("# ").strip()
                await self.client.write_file(file_path, content.strip())
                applied = True

        if applied:
            return "Applied changes from code blocks in response"

        return f"Could not parse structured response. Raw response: {response[:500]}"