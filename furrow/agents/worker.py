from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        # Gather current contents of files the task touches
        file_contents: dict[str, str] = {}
        for file_path in self.task.files:
            try:
                file_contents[file_path] = await self.client.read_file(file_path)
            except FileNotFoundError:
                file_contents[file_path] = ""  # new file to be created

        if not file_contents:
            # No specific files: discover workspace files for context
            for file_path in self.client.list_files(".")[:20]:
                try:
                    file_contents[file_path] = await self.client.read_file(file_path)
                except Exception:  # noqa: BLE001, S112 - skip unreadable discovery files
                    continue

        files_section = ""
        if file_contents:
            files_section = "\n".join(
                f"--- {path} ---\n{content}" for path, content in file_contents.items()
            )

        prompt = f"""{WORKER_PROMPT}

Task: {self.task.description}

Current file contents:
{files_section if files_section else "(no existing files)"}

Return JSON only (no markdown) with this shape:
{{
  "changes": [
    {{
      "file": "path/to/file.py",
      "content": "full new file content"
    }}
  ],
  "summary": "Brief summary of changes made"
}}

Write the complete new content for each file. Create new files if needed."""

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            for change in changes:
                await self.client.write_file(change["file"], change["content"])
            summary = data.get("summary", f"Applied changes to {len(changes)} file(s)")
            return summary
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            # Fallback: return raw response if JSON parsing fails
            return response.strip()
