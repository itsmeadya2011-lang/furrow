from __future__ import annotations

import json
import re

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings or Settings())
        self.settings = self.client.settings

    async def run(self) -> str:
        # First, read the files that need to be modified
        file_contents = {}
        for file_path in self.task.files:
            content = await self.client.read_file(self.settings.workspace / file_path)
            file_contents[file_path] = content

        # Build the prompt with file context
        prompt = self._build_prompt(file_contents)

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse the response for file changes and apply them
        return await self._apply_changes(response, file_contents)

    def _build_prompt(self, file_contents: dict[str, str]) -> str:
        files_context = ""
        if file_contents:
            files_context = "\n\n=== Existing Files ===\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content}\n"

        files_to_touch = ", ".join(self.task.files) if self.task.files else "any relevant files"

        return f"""{WORKER_PROMPT}

Task: {self.task.description}
Files to modify: {files_to_touch}
{files_context}

=== Instructions ===
Return your changes as JSON in this format:
{{
  "changes": [
    {{
      "file": "path/to/file.py",
      "content": "full new file content"
    }}
  ],
  "summary": "Brief description of what was changed"
}}

If no changes needed, return: {{"changes": [], "summary": "No changes required"}}
"""

    async def _apply_changes(self, response: str, original_files: dict[str, str]) -> str:
        try:
            # Try to parse JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
        except json.JSONDecodeError:
            # If not JSON, return the response as summary
            return response

        changes = data.get("changes", [])
        summary = data.get("summary", "Task completed")

        if not changes:
            return summary

        for change in changes:
            file_path = change.get("file")
            content = change.get("content")

            if not file_path or content is None:
                continue

            full_path = self.settings.workspace / file_path

            # Write the file
            await self.client.write_file(full_path, content)

            # Track in task files if not already there
            if file_path not in self.task.files:
                self.task.files.append(file_path)

        return f"{summary}\nModified files: {', '.join(c['file'] for c in changes)}"
