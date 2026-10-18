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
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        # Read existing file contents for context
        file_contents: dict[str, str] = {}
        for path in self.task.files:
            try:
                file_contents[path] = await self.client.read_file(path)
            except FileNotFoundError:
                file_contents[path] = ""  # new file
            except Exception:
                file_contents[path] = ""

        context = ""
        if file_contents:
            context = "\n\nExisting file contents:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}"
            f"{context}\n\n"
            "Return JSON only with this shape:\n"
            "{\n"
            '  "files": {\n'
            '    "path/to/file.py": "full file content"\n'
            "  },\n"
            '  "summary": "brief description of changes"\n'
            "}\n"
            "Include every file you are creating or modifying. "
            "Omit files you are not changing.\n"
        )

        response = await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            match = re.search(r"```json\s*\n(.*?)\n```", response, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
            else:
                # Fallback: treat entire response as a summary, no file changes
                return response.strip()

        files = data.get("files", {})
        summary = data.get("summary", "No summary provided")

        for path, content in files.items():
            await self.client.write_file(path, content)

        return summary
