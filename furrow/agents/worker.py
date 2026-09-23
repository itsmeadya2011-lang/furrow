from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)

WORKER_PROMPT_WITH_FILES = """{base_prompt}

You must return your response as JSON only (no markdown, no explanation) with this exact structure:
{{
  "edits": [
    {{"file": "path/to/file", "content": "full new file content"}},
    ...
  ],
  "summary": "Brief description of changes"
}}

If no edits are needed, return {{"edits": [], "summary": "..."}}."""


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        file_contents = {}
        for file_path in self.task.files or []:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                logger.warning("File not found, skipping: %s", file_path)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to read %s: %s", file_path, e)

        files_section = ""
        if file_contents:
            files_section = "\n\nFile contents:\n"
            for path, content in file_contents.items():
                files_section += f"\n--- {path} ---\n{content}\n"

        prompt = WORKER_PROMPT_WITH_FILES.format(base_prompt=WORKER_PROMPT)
        prompt += f"\n\nTask: {self.task.description}"
        if self.task.files:
            prompt += f"\nFiles to touch: {', '.join(self.task.files)}"
        prompt += files_section

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            parsed = json.loads(response.strip())
            edits = parsed.get("edits", [])
            summary = parsed.get("summary", "No summary provided")
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, returning raw text")
            return response

        for edit in edits:
            file_path = edit.get("file")
            content = edit.get("content", "")
            if file_path:
                try:
                    await self.client.write_file(file_path, content)
                except Exception as e:  # noqa: BLE001
                    logger.error("Failed to write %s: %s", file_path, e)

        edit_count = len(edits)
        return f"{summary} ({edit_count} file{'s' if edit_count != 1 else ''} edited)"
