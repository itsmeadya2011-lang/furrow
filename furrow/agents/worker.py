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
        # First, read any relevant files to provide context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except Exception:
                # File might not exist yet
                pass

        context = ""
        if file_contents:
            context = "\n\nExisting file contents:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{context}\n"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Parse the response for file operations
        # Expected format: JSON with files_modified array containing {path, content}
        try:
            data = json.loads(response)
            if "files_modified" in data and isinstance(data["files_modified"], list):
                for file_mod in data["files_modified"]:
                    if isinstance(file_mod, dict) and "path" in file_mod and "content" in file_mod:
                        await self.client.write_file(file_mod["path"], file_mod["content"])
                return data.get("summary", "Task completed with file modifications")
            return data.get("summary", response)
        except (json.JSONDecodeError, ValueError):
            # Try to extract file modifications from markdown code blocks
            return await self._parse_and_apply_markdown(response)

    async def _parse_and_apply_markdown(self, response: str) -> str:
        """Parse markdown code blocks and apply file changes."""
        # Pattern: ```python\n# filepath: path/to/file\ncontent\n```
        pattern = r"```(?:\w+)?\n# filepath: (.+?)\n(.*?)\n```"
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            for file_path, content in matches:
                file_path = file_path.strip()
                content = content.strip()
                await self.client.write_file(file_path, content)
            return f"Applied {len(matches)} file modification(s) from markdown blocks"
        
        # If no structured format found, return the raw response
        return response
