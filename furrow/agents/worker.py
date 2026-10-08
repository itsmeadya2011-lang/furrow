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
        # Read the files listed in task.files using client.read_file()
        file_contents = {}
        if self.task.files:
            for filepath in self.task.files:
                try:
                    content = await self.client.read_file(filepath)
                    file_contents[filepath] = content
                except Exception:
                    file_contents[filepath] = None

        # Build a prompt that includes the current file contents
        files_info = ""
        for filepath, content in file_contents.items():
            files_info += f"\n\n=== {filepath} ===\n"
            if content is None:
                files_info += "(File could not be read)\n"
            else:
                files_info += content

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"{files_info}\n"
        )

        # Call the LLM to get the changes
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse the JSON response to extract file changes
        response = self._strip_json_markers(response)

        try:
            data = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            return f"Failed to parse worker response: {response}"

        # Write the changes using client.write_file()
        files = data.get("files", {})
        for filepath, content in files.items():
            if isinstance(content, str):
                try:
                    await self.client.write_file(filepath, content)
                except Exception as e:
                    return f"Failed to write {filepath}: {e}"

        summary = data.get("summary", "Changes applied")
        return summary

    def _strip_json_markers(self, text: str) -> str:
        text = text.strip()
        # Remove markdown code fences
        if text.startswith("```"):
            # Find the closing fence
            lines = text.splitlines()
            if lines[0].startswith("```"):
                # Remove first line if it's just ```json or ```
                first_line = lines[0].strip()
                if first_line in ("```", "```json", "```JSON"):
                    lines = lines[1:]
                else:
                    lines = lines[1:]  # Just in case
            # Remove last line if it's closing fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text