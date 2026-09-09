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
        file_contexts: list[str] = []
        for path in self.task.files:
            try:
                content = await self.client.read_file(path)
                file_contexts.append(f"File: {path}\n```\n{content}\n```")
            except Exception:
                pass

        files_section = "\n\n".join(file_contexts) if file_contexts else "No existing files provided."
        files_list = ", ".join(self.task.files) if self.task.files else "any relevant files"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_list}\n\n"
            f"Current file contents:\n{files_section}\n\n"
            "Return ONLY a JSON object with this shape:\n"
            '{"files": [{"path": "relative/path.py", "content": "full new file content"}], "summary": "what was changed"}'
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        summary = "Worker completed task."

        try:
            data = json.loads(response)
            for file_entry in data.get("files", []):
                path = file_entry.get("path")
                content = file_entry.get("content", "")
                if path:
                    await self.client.write_file(path, content)
            summary = data.get("summary", summary)
        except (json.JSONDecodeError, AttributeError):
            sections = re.split(r"\n(?=###\s|\n###\s)", response)
            for section in sections:
                lines = section.strip().splitlines()
                if not lines:
                    continue
                path_match = re.match(r"^###\s+(.+)$", lines[0])
                if not path_match:
                    continue
                path = path_match.group(1).strip()
                content_parts: list[str] = []
                in_code = False
                for line in lines[1:]:
                    if re.match(r"^```", line):
                        in_code = not in_code
                        continue
                    if in_code:
                        content_parts.append(line)
                if path and content_parts:
                    await self.client.write_file(path, "\n".join(content_parts))
            summary = response[:200] if response else summary

        return summary
