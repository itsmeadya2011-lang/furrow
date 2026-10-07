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
        context_parts = []
        for filepath in self.task.files:
            try:
                content = await self.client.read_file(filepath)
                context_parts.append(f"--- {filepath} ---\n{content}\n")
            except FileNotFoundError:
                context_parts.append(f"--- {filepath} ---\n(FILE NOT FOUND - will create)\n")
            except Exception as e:
                context_parts.append(f"--- {filepath} ---\n(ERROR reading: {e})\n")

        context = "\n".join(context_parts) if context_parts else "(no existing files to reference)"

        prompt = f"""{WORKER_PROMPT}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}

Existing file contents:
{context}

Return a JSON object with this shape:
{{
  "files": {{
    "path/to/file.py": "full file content here"
  }},
  "summary": "concise summary of what you changed"
}}

Only include files that need to be created or modified. Return valid JSON only, no markdown.
"""

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    return f"Failed to parse worker response as JSON. Raw response: {response[:500]}"
            else:
                return f"Failed to parse worker response as JSON. Raw response: {response[:500]}"

        files_written = []
        for filepath, content in data.get("files", {}).items():
            await self.client.write_file(filepath, content)
            files_written.append(filepath)

        summary = data.get("summary", "Task completed")
        return f"Completed task. Files written: {', '.join(files_written) if files_written else 'none'}. Summary: {summary}"
