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
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        return await self._parse_and_write(response)

    async def _parse_and_write(self, response: str) -> str:
        written = []
        errors = []

        # Try JSON format first
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "files" in data:
                for item in data["files"]:
                    if isinstance(item, dict) and "path" in item and "content" in item:
                        try:
                            await self.client.write_file(item["path"], item["content"])
                            written.append(item["path"])
                        except OSError as e:
                            errors.append(f"{item['path']}: {e}")
                summary = data.get("summary", f"Wrote {len(written)} file(s)")
                return self._format_result(summary, written, errors)
        except json.JSONDecodeError:
            pass

        # Fall back to markdown code blocks with file path hints
        # Pattern: ```language:path/to/file or ```path/to/file
        pattern = r"```(?:[\w+.-]+:)?([^\n]+)\n([\s\S]*?)```"
        matches = re.findall(pattern, response)
        for path, content in matches:
            path = path.strip()
            if path and content is not None:
                try:
                    await self.client.write_file(path, content.rstrip() + "\n")
                    written.append(path)
                except OSError as e:
                    errors.append(f"{path}: {e}")

        if written:
            return self._format_result(f"Wrote {len(written)} file(s) from code blocks", written, errors)

        # If nothing parsed, return the raw response as summary
        return f"Response (no files written): {response[:500]}"

    def _format_result(self, summary: str, written: list[str], errors: list[str]) -> str:
        parts = [summary]
        if written:
            parts.append("Files written:")
            parts.extend(f"  - {f}" for f in written)
        if errors:
            parts.append("Errors:")
            parts.extend(f"  - {e}" for e in errors)
        return "\n".join(parts)
